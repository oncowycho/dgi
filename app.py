import streamlit as st
import pydicom
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from skimage.measure import find_contours
import plotly.graph_objects as go
import base64
from six import iteritems
from loess.loess_1d import loess_1d
from multiprocessing import Pool
from pydicom.uid import ImplicitVRLittleEndian
from dicompylercore import dicomparser, dvh
import matplotlib.path
import plotly.express as px

st.set_page_config(layout="wide",page_title = "Dose Gradient Curve Analyzer", page_icon="dgi_tab.ico",)
theme = st.get_option("theme.base")

if theme == "dark":
    st.logo('logo_light.png')
else:
    st.logo('logo_dark.png')

st.markdown(f"""
    <style>
    /* Hide the configuration menu (three dots) */
    [data-testid="stToolbar"] {{
        visibility: hidden;
        height: 0px;
    }}
    </style>
""", unsafe_allow_html=True)


def get_dvh(rtss, rtdose, roi, limit=None, callback=None):
    """Calculate a cumulative DVH in Gy from a DICOM RT Structure Set & Dose."""
    structures = rtss.GetStructures()

    s = structures[roi]
    s['planes'] = rtss.GetStructureCoordinates(roi)
    
    s['thickness'] = rtss.CalculatePlaneThickness(s['planes'])
    hist = calculate_dvh(s, rtdose, limit, callback)
    return dvh.DVH(counts=hist,
                   bins=(np.arange(0, 2) if (hist.size == 1) else
                         np.arange(0, hist.size + 1) / 100),
                   dvh_type='differential',
                   dose_units='gy',
                   name=s['name']
                   ).cumulative


def calculate_dvh(structure, dose, limit=None, callback=None):
    """Calculate the differential DVH for the given structure and dose grid."""
    planes = structure['planes']

    if ((len(planes)) and ("PixelData" in dose.ds)):
        dd = dose.GetDoseData()
        id = dose.GetImageData()

        x, y = np.meshgrid(np.array(dd['lut'][0]), np.array(dd['lut'][1]))
        x, y = x.flatten(), y.flatten()
        dosegridpoints = np.vstack((x, y)).T

        maxdose = int(dd['dosemax'] * dd['dosegridscaling'] * 100)
        if isinstance(limit, int):
            if (limit < maxdose):
                maxdose = limit
        hist = np.zeros(maxdose)
    else:
        return np.array([0])

    n = 0
    planedata = {}
    for z, plane in iteritems(planes):
        doseplane = dose.GetDoseGrid(z)
        planedata[z] = calculate_plane_histogram(
            plane, doseplane, dosegridpoints,
            maxdose, dd, id, structure, hist)
        n += 1
        if callback:
            callback(n, len(planes))
    volume = sum([p[1] for p in planedata.values()]) / 1000
    hist = sum([p[0] for p in planedata.values()])
    hist = hist * volume / sum(hist)
    hist = np.trim_zeros(hist, trim='b')

    return hist


def calculate_plane_histogram(plane, doseplane, dosegridpoints, maxdose, dd, id, structure, hist):
    contours = [[x[0:2] for x in c['data']] for c in plane]

    if not len(doseplane):
        return (np.arange(0, maxdose), 0)

    grid = np.zeros((dd['rows'], dd['columns']), dtype=np.uint8)

    for i, contour in enumerate(contours):
        m = get_contour_mask(dd, id, dosegridpoints, contour)
        grid = np.logical_xor(m.astype(np.uint8), grid).astype(bool)

    hist, vol = calculate_contour_dvh(
        grid, doseplane, maxdose, dd, id, structure)
    return (hist, vol)


def get_contour_mask(dd, id, dosegridpoints, contour):
    doselut = dd['lut']

    c = matplotlib.path.Path(list(contour))
    grid = c.contains_points(dosegridpoints)
    grid = grid.reshape((len(doselut[1]), len(doselut[0])))

    return grid


def calculate_contour_dvh(mask, doseplane, maxdose, dd, id, structure):
    mask = np.ma.array(doseplane * dd['dosegridscaling'] * 100, mask=~mask)
    hist, edges = np.histogram(mask.compressed(),
                               bins=maxdose,
                               range=(0, maxdose))

    vol = sum(hist) * ((id['pixelspacing'][0]) *
                       (id['pixelspacing'][1]) *
                       (structure['thickness']))
    return hist, vol


def read_dose_dicom(dicom_file_path):
    ds = pydicom.dcmread(dicom_file_path, force=True)
    if not hasattr(ds.file_meta, 'TransferSyntaxUID'):
        ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    dose_grid_scaling = ds.DoseGridScaling
    dose_data = ds.pixel_array * dose_grid_scaling
    ipp = np.array(ds.ImagePositionPatient).astype(float)
    pixel_spacing = np.array(ds.PixelSpacing).astype(float)
    grid_frame_offset_vector = np.array(ds.GridFrameOffsetVector).astype(float) if 'GridFrameOffsetVector' in ds else [0]
    return dose_data, ipp, pixel_spacing, grid_frame_offset_vector


def process_slice(args):
    z, dose_slice, ipp, pixel_spacing, grid_frame_offset_vector, dose_values = args
    coordinates = []
    z_coord = ipp[2] + grid_frame_offset_vector[z]
    for dose in dose_values:
        contours = find_contours(dose_slice, dose)
        if len(contours) == 0:
            continue
        for coords in contours:
            for y_index, x_index in coords:
                x_coord = ipp[0] + x_index * pixel_spacing[1]
                y_coord = ipp[1] + y_index * pixel_spacing[0]
                coordinates.append([x_coord, y_coord, z_coord, dose])
    return coordinates


def extract_dose_coordinates_parallel(dose_data, ipp, pixel_spacing, grid_frame_offset_vector, prescript_dose, min_dose, step_size, step_type):
    vmin = max(round(np.min(dose_data), -1), min_dose)
    vmax = np.max(dose_data)

    dose_values = np.append(np.sort(np.arange(prescript_dose,vmin,-step_size)),np.arange(prescript_dose+step_size,vmax,step_size))

    with Pool() as pool:
        results = pool.map(process_slice, [(z, dose_data[z], ipp, pixel_spacing, grid_frame_offset_vector, dose_values) for z in range(dose_data.shape[0])])
    coordinates = [item for sublist in results for item in sublist]
    return coordinates


def extract_dose_coordinates(dose_data, ipp, pixel_spacing, grid_frame_offset_vector, prescript_dose, min_dose, step_size, step_type):
    mask = dose_data >= min_dose
    coords = np.argwhere(mask)
    doses = dose_data[mask]
   
    vmin = max(round(np.min(dose_data), -1), min_dose)
    vmax = np.max(dose_data)

    dose_values = np.append(np.sort(np.arange(prescript_dose,vmin,-step_size)),np.arange(prescript_dose+step_size,vmax,step_size))
    
    coordinates = []
    for z in range(dose_data.shape[0]):
        z_coord = ipp[2] + grid_frame_offset_vector[z]
        for dose in dose_values:
            contours = find_contours(dose_data[z], dose)
            if len(contours) == 0:
                continue
            for coords in contours:
                for y_index, x_index in coords:
                    x_coord = ipp[0] + x_index * pixel_spacing[1]
                    y_coord = ipp[1] + y_index * pixel_spacing[0]
                    coordinates.append([x_coord, y_coord, z_coord, dose])
    return coordinates


def calculate_dgi(dose_coordinates, min_dose, prescript_dose, step_size, step_type):
    df = pd.DataFrame(dose_coordinates, columns=["x", "y", "z", "dose"]).sort_values(by='dose')
    df = df[df.dose >= min_dose]
    df = df.drop_duplicates()

    df['dose'] = df['dose'].astype('category')
    grouped = df['dose'].cat.categories

    dgi_para = []
    area0, volume0 = 0, 0
    cDGI = None

    for dose in grouped[::-1]:
        points = df[df['dose'] == dose][['x', 'y', 'z']].values

        if len(points) >= 4:
            hull = ConvexHull(points, qhull_options='QJ')
            area, volume = hull.area, hull.volume
        else:
            continue

        dDGI = (volume - volume0) / (0.5 * (area + area0))
        dose = round(dose,3)
        pdose = (dose / prescript_dose) * 100

        if dose >= min_dose and dose != grouped[-1]:
            if   dose  < prescript_dose: cDGI += dDGI
            elif dose == prescript_dose: cDGI = 0

            dgi_para.append([dose, pdose, area, volume, dDGI, cDGI])

        if dose != grouped[-1]: area0, volume0 = area, volume
            
    return pd.DataFrame(dgi_para, columns=["Dose", "Dose (%)","Area", "Volume", "dDGI", "cDGI"])


def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="dgi_parameters.csv">Download dgi_parameters.csv</a>'
    return href

def make_vivid(color):
    r, g, b = color
    # Set minimum or maximum RGB values to make the color vivid
    r = max(min(r, 255), 200) if r > 128 else min(max(r, 0), 50)
    g = max(min(g, 255), 200) if g > 128 else min(max(g, 0), 50)
    b = max(min(b, 255), 200) if b > 128 else min(max(b, 0), 50)
    return r, g, b

def make_vivid_and_contrasting(color):
    r, g, b = color
    # Calculate brightness as the average of RGB values
    brightness = (r + g + b) / 3

    # Exclude colors that are too bright (e.g., average brightness above 200)
    if brightness > 200:
        # If color is too bright, adjust to a vivid darker color
        r = min(r, 180)
        g = min(g, 180)
        b = min(b, 180)
    
    # Ensure each component is either close to 0 or high for saturation
    r = max(min(r, 255), 200) if r > 128 else min(max(r, 0), 50)
    g = max(min(g, 255), 200) if g > 128 else min(max(g, 0), 50)
    b = max(min(b, 255), 200) if b > 128 else min(max(b, 0), 50)
    
    return r, g, b


def cal_dvh(fig, rtss, rtdose, RTstructures, structure_name_to_id, selected_structure_names, unit, dunit):
    calcdvhs = {}
    
    # Loop through the selected structure names and get the corresponding structure IDs
    for structure_name in selected_structure_names:
        structure_id = structure_name_to_id[structure_name]
        structure = RTstructures[structure_id]
        vivid_color = make_vivid(structure["color"])
        vivid_contrast_color = make_vivid_and_contrasting(structure["color"])
        color_string = f'rgb({vivid_contrast_color[0]}, {vivid_contrast_color[1]}, {vivid_contrast_color[2]})'
        
        calcdvhs[structure_id] = get_dvh(rtss, rtdose, structure_id)

        if calcdvhs[structure_id].counts.any():
            # Add the DVH plot for each selected structure
            fig.add_trace(go.Scatter(
                x=np.arange(0, len(calcdvhs[structure_id].counts), dtype=float) / dunit,  # Ensure float division
                y=calcdvhs[structure_id].counts * 100 / calcdvhs[structure_id].counts[0],
                mode='lines',
                name=structure['name'],
                line=dict(color=color_string, dash='dash', width=3),
                yaxis='y'
            ))

    return fig
    

def setup_page_style():
    """Setup initial page styling and header"""
    st.markdown("""
        <style>
        .sticky-header {
            top: 0;
            left: 0;
            background-color: white;
            z-index: 999;
            padding: 10px 0;
            margin: 0;
            width: 100%;
        }
        
        .title-text {
            text-align: center;
            margin: 0;
            padding: 0;
            font-size: 42px;  /* Reduced from 48px */
            font-weight: normal;  /* Explicitly set font weight */
            color: black;
            font-family: "Arial Black", sans-serif;  /* Changed from "Arial Black" */
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        }

        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f1f1f1;
            padding: 10px;
            text-align: center;
            z-index: 999;
            font-size: 14px;
            color: #333;
        }

        .main-content {
            margin-top: 80px;     /* Add space below fixed header */
            margin-bottom: 50px;  /* Space for footer */
            padding-top: 0px;
        }

        /* Remove ALL default Streamlit spacing */
        .block-container {
            padding-left: 10rem !important;
            padding-right: 10rem !important;
            padding-bottom: 0rem !important;
            padding-top: 1rem !important;
        }

        /* Remove default Streamlit header padding and add side margins */
        .appview-container .main .block-container {
            padding-top: 0rem;
            padding-right: 5rem;
            padding-left: 5rem;
            padding-bottom: 1rem;
        }

        /* Hide Streamlit's default header */
        header[data-testid="stHeader"] {
            display: none;
        }

        /* Adjust top section padding */
        section[data-testid="stSidebar"] > div {
            padding-top: 0rem;
        }

        section.main > div:first-child {
            padding-top: 0rem;
            padding-right: 5rem;
            padding-left: 5rem;
        }
        </style>
        
        <div class="sticky-header">
            <h1 class="title-text">Dose Gradient Curve Analyzer</h1>
        </div>
        <div class="footer">
            © 2024 Wonyoung Cho. All rights reserved. |
            Contact: <a href="mailto:wycho@oncosoft.io" style="text-decoration: none; color: DodgerBlue;">
            wycho@oncosoft.io</a>
        </div>  
        <div class="main-content">
    """, unsafe_allow_html=True)
    
def get_user_inputs():
    """Get all user inputs from sidebar"""
    uploaded_file = st.sidebar.file_uploader("Upload RT Dose (dose.dcm)", type=["dcm"])
    uploaded_structure_file = st.sidebar.file_uploader("Upload RT Structure (rts.dcm) (Optional)", type=["dcm"])
    
    show_isodose = st.sidebar.toggle('Show Isodose View', value=False)

    prescript_dose = st.sidebar.number_input('Prescription Dose (Gy)', min_value=0.0, value=40.0, format="%.2f")
    min_dose = st.sidebar.number_input('Minimum Dose (Gy)', min_value=0.1, value=0.1, step=0.1, format="%.2f")
    step_type = st.sidebar.radio('Dose step size',['Absolute (Gy)', 'Relative (%)'], horizontal=True)
    
    step = 0.1
    fmt = '%.2f'
    
    # Store previous values in session state if they don't exist
    if 'prev_step_size' not in st.session_state:
        st.session_state.prev_step_size = 1.0
    if 'prev_step_type' not in st.session_state:
        st.session_state.prev_step_type = step_type
    if 'prev_prescript_dose' not in st.session_state:
        st.session_state.prev_prescript_dose = prescript_dose
    if 'prev_min_dose' not in st.session_state:
        st.session_state.prev_min_dose = min_dose
        
    step_size = round(st.sidebar.number_input('Step Size', 
                                            min_value=step, 
                                            max_value=9.0, 
                                            value=st.session_state.prev_step_size, 
                                            step=step, 
                                            format=fmt,
                                            label_visibility="collapsed"), 3)
    
    # Check if any parameter has changed
    if (step_size != st.session_state.prev_step_size or 
        step_type != st.session_state.prev_step_type or
        prescript_dose != st.session_state.prev_prescript_dose or
        min_dose != st.session_state.prev_min_dose):
        # Clear processed data to force recalculation
        st.session_state.processed_data = None
        
        # Update previous values
        st.session_state.prev_step_size = step_size
        st.session_state.prev_step_type = step_type
        st.session_state.prev_prescript_dose = prescript_dose
        st.session_state.prev_min_dose = min_dose

    return uploaded_file, uploaded_structure_file, prescript_dose, min_dose, step_type, step_size, show_isodose

def process_dicom_files(uploaded_file, uploaded_structure_file):
    """Process uploaded DICOM files and return required objects"""
    
    if not uploaded_file: 
        dicom_file = 'dose.dcm'
    else:
        dicom_file = uploaded_file.name
        with open(dicom_file, "wb") as f:   
            f.write(uploaded_file.getbuffer())

    if not uploaded_structure_file:
        structure_file = 'rts.dcm'
    else:
        structure_file = uploaded_structure_file.name
        with open(structure_file, "wb") as f:
            f.write(uploaded_structure_file.getbuffer())

    rtdose = None
    rtss = None
    RTstructures = None
    structure_name_to_id = None

    rtdose_file = pydicom.dcmread(dicom_file, force=True)
    if not hasattr(rtdose_file.file_meta, 'TransferSyntaxUID'):
        rtdose_file.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    rtdose = dicomparser.DicomParser(rtdose_file)

    rtss_file = pydicom.dcmread(structure_file, force=True)
    
    # Check if StudyInstanceUIDs match
    if hasattr(rtss_file, 'StudyInstanceUID') and hasattr(rtdose_file, 'StudyInstanceUID'):
        if rtss_file.StudyInstanceUID == rtdose_file.StudyInstanceUID:
            rtss = dicomparser.DicomParser(rtss_file)
            RTstructures = rtss.GetStructures()
            structure_name_to_id = {structure['name']: key for key, structure in RTstructures.items()}
        else:
            st.warning("StudyInstanceUID mismatch between dose and structure files.")

    return dicom_file, rtdose, rtss, RTstructures, structure_name_to_id

def setup_plot_parameters(step_type, prescript_dose):
    """Setup plotting parameters based on step type"""
    unit = ' (Gy)' if step_type == 'Absolute (Gy)' else ''
    
    if step_type == 'Absolute (Gy)':
        xidx = 'Dose'
        prescript = prescript_dose
        dunit = 100
        dtick = 2
    else:
        xidx = 'Dose (%)'
        prescript = 100
        dunit = prescript_dose
        dtick = 5
        
    return unit, xidx, prescript, dunit, dtick

def create_2d_isodose_plot(dose_data, ipp, pixel_spacing, grid_frame_offset_vector, prescript_dose, step_size, min_dose):
    """Create 2D isodose view with slice slider"""
    
    # Initialize session state for slice index if it doesn't exist
    if 'slice_idx' not in st.session_state:
        st.session_state.slice_idx = dose_data.shape[0]//2
    
    # Create a row for slider controls
    col1, col2, col3 = st.columns([1, 8, 1])
    
    # Previous slice button
    with col1:
        if st.button('←'):
            st.session_state.slice_idx = max(0, st.session_state.slice_idx - 1)
    
    # Slice slider
    with col2:
        st.session_state.slice_idx = st.slider(
            'Slice',
            0,
            dose_data.shape[0]-1,
            st.session_state.slice_idx,
            key='slice_slider'
        )
    
    # Next slice button
    with col3:
        if st.button('→'):
            st.session_state.slice_idx = min(dose_data.shape[0]-1, st.session_state.slice_idx + 1)
    
    # Get current slice data using session state
    slice_data = dose_data[st.session_state.slice_idx]
    z_coord = ipp[2] + grid_frame_offset_vector[st.session_state.slice_idx]
    
    # Create contour plot
    fig = go.Figure()
    
    # Calculate contour levels based on step size
    max_dose = np.max(slice_data)
    min_dose_data = np.min(slice_data)  # Use actual minimum from data
    levels = np.arange(min_dose_data, max_dose + step_size, step_size)  # Start from actual minimum
    
    # Add contour plot
    fig.add_trace(go.Contour(
        z=slice_data,
        x=np.arange(slice_data.shape[1]) * pixel_spacing[1] + ipp[0],
        y=np.arange(slice_data.shape[0]) * pixel_spacing[0] + ipp[1],
        contours=dict(
            start=min_dose_data,
            end=max_dose,
            size=step_size,
            showlabels=True,
            labelfont=dict(size=12, color='white')
        ),
        colorscale='Viridis',
        colorbar=dict(
            title=dict(
                text='Dose (Gy)',
                font=dict(size=14, family="Arial Black")
            ),
            tickfont=dict(size=12, family="Arial Black"),
            x=1.02,  # Position colorbar closer to plot
            y=1,     # Align with top
            yanchor='top',  # Anchor to top
            len=1,   # Full length
            thickness=20    # Adjust thickness of colorbar
        ),
        hoverongaps=False,
        hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<br>Dose: %{z:.1f} Gy<extra></extra>'
    ))
    
    # Highlight minimum dose (user specified)
    min_contours = find_contours(slice_data, min_dose)
    for contour in min_contours:
        x_coords = contour[:, 1] * pixel_spacing[1] + ipp[0]
        y_coords = contour[:, 0] * pixel_spacing[0] + ipp[1]
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines',
            line=dict(color='blue', width=2),
            name=f'Minimum ({min_dose} Gy)',
            hoverinfo='skip'
        ))
    
    # Highlight prescription dose
    prescript_contours = find_contours(slice_data, prescript_dose)
    for contour in prescript_contours:
        x_coords = contour[:, 1] * pixel_spacing[1] + ipp[0]
        y_coords = contour[:, 0] * pixel_spacing[0] + ipp[1]
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='lines',
            line=dict(color='red', width=2),
            name=f'Prescription ({prescript_dose} Gy)',
            hoverinfo='skip'
        ))
    
    # Update layout with fixed dimensions and no auto-sizing
    fig.update_layout(
        title={
            'text': f'Isodose View (Z = {z_coord:.1f} mm)',
            'font': dict(size=22, family="Arial Black", color="black"),
            'y': 0.98,
            'yanchor': 'top',
            'pad': dict(t=0)
        },
        xaxis_title='X (mm)',
        yaxis_title='Y (mm)',
        xaxis=dict(
            title_font=dict(size=18, family="Arial Black", color="black"),
            tickfont=dict(size=14, family="Arial Black", color="black"),
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            title_font=dict(size=18, family="Arial Black", color="black"),
            tickfont=dict(size=14, family="Arial Black", color="black")
        ),
        font=dict(family="Arial Black", size=18, color="black"),
        legend=dict(
            x=1.12,
            y=1,
            xanchor='left',
            yanchor='top',
            font=dict(size=12),
            bgcolor='rgba(255, 255, 255, 0.7)',
            bordercolor='rgba(0, 0, 0, 0.3)',
            borderwidth=1
        ),
        showlegend=True,
        width=800,         # Fixed width
        height=800,        # Fixed height
        margin=dict(
            l=80,    # Left margin
            r=150,   # Right margin
            t=50,    # Top margin
            b=80     # Bottom margin
        ),
        plot_bgcolor='white',
        autosize=False    # Disable autosize
    )
    
    st.plotly_chart(fig, use_container_width=True, config={
        'displayModeBar': True,
        'responsive': True    # Disable responsive resizing
    })
    # Add JavaScript for keyboard controls
    st.markdown("""
        <script>
        document.addEventListener('keydown', function(e) {
            if (e.key === 'ArrowUp') {
                // Increment slice
                window.streamlitPyRef.setComponentValue('slice_slider', 
                    Math.min(window.streamlitPyRef.getComponentValue('slice_slider') + 1, maxSlice));
            } else if (e.key === 'ArrowDown') {
                // Decrement slice
                window.streamlitPyRef.setComponentValue('slice_slider', 
                    Math.max(window.streamlitPyRef.getComponentValue('slice_slider') - 1, 0));
            }
        });
        </script>
    """, unsafe_allow_html=True)

def create_dgi_plots(dgi_parameters, xidx, prescript, dtick, unit, dunit, dose_data, ipp, pixel_spacing, grid_frame_offset_vector, step_size, min_dose, show_isodose, rtss=None, rtdose=None, RTstructures=None, structure_name_to_id=None, selected_structure_names=None):
    """Create and return dDGI, cDGI, and 2D isodose plots"""
    # Create interpolated parameters
    dgi_parameters_new = interpolate_dgi_parameters(dgi_parameters)
    
    # Create dDGI plot and get max_x and min_dDGI_x
    fig_ddgi, max_x, min_dDGI_x = create_ddgi_plot(dgi_parameters_new, xidx, prescript, dtick, unit)
    
    # Create cDGI plot with max_x and min_dDGI_x
    fig_cdgi = create_cdgi_plot(dgi_parameters_new, xidx, dtick, unit, prescript, max_x, min_dDGI_x)
    
    # Add DVH if structure file exists
    if rtss and rtdose and RTstructures and structure_name_to_id and selected_structure_names:
        fig_cdgi = cal_dvh(fig_cdgi, rtss, rtdose, RTstructures, structure_name_to_id, selected_structure_names, unit, dunit)
    
    # Create 2D isodose plot only if toggle is on
    if show_isodose:
        create_2d_isodose_plot(dose_data, ipp, pixel_spacing, grid_frame_offset_vector, prescript, step_size, min_dose)
    
    return fig_ddgi, fig_cdgi

def interpolate_dgi_parameters(dgi_parameters):
    """Create interpolated DGI parameters"""
    # Create new dose points with interval of 1
    dose_new = np.arange(np.floor(dgi_parameters['Dose'].min()), 
                       np.ceil(dgi_parameters['Dose'].max()) + 1, 1)
    
    # Interpolate all columns
    dose_pct_new = np.interp(dose_new, np.flip(dgi_parameters['Dose']), np.flip(dgi_parameters['Dose (%)']))
    area_new = np.interp(dose_new, np.flip(dgi_parameters['Dose']), np.flip(dgi_parameters['Area']))
    volume_new = np.interp(dose_new, np.flip(dgi_parameters['Dose']), np.flip(dgi_parameters['Volume']))
    ddgi_new = np.interp(dose_new, np.flip(dgi_parameters['Dose']), np.flip(dgi_parameters['dDGI']))
    cdgi_new = np.interp(dose_new, np.flip(dgi_parameters['Dose']), np.flip(dgi_parameters['cDGI']))

    return pd.DataFrame({
        'Dose': np.flip(dose_new),
        'Dose (%)': np.flip(dose_pct_new),
        'Area': np.flip(area_new),
        'Volume': np.flip(volume_new),
        'dDGI': np.flip(ddgi_new),
        'cDGI': np.flip(cdgi_new)
    })

def create_ddgi_plot(dgi_parameters_new, xidx, prescript, dtick, unit):
    """Create dDGI plot and return min dDGI x-value"""
    x_dgi = dgi_parameters_new[xidx]
    y_dgi = dgi_parameters_new['dDGI']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_dgi, y=y_dgi, mode='markers', marker=dict(color='royalblue'), name='dDGI'))
    
    xout, yout, wout = loess_1d(x_dgi.values, y_dgi.values, frac=.2)
    fig.add_trace(go.Scatter(x=xout, y=yout, mode='lines', marker=dict(color='lightskyblue'), name='Regression'))

    # Find minimum dDGI point
    range_around_prescript = 0.1 * prescript
    nearby_points = dgi_parameters_new[
        (x_dgi >= prescript - range_around_prescript) & 
        (x_dgi <= prescript + range_around_prescript)
    ]

    min_dDGI_x = None
    if not nearby_points.empty:
        min_dDGI = nearby_points["dDGI"].min()
        min_dDGI_idx = nearby_points["dDGI"].idxmin()
        min_dDGI_x = nearby_points[xidx].loc[min_dDGI_idx]  
        fig.add_trace(go.Scatter(
            x=[min_dDGI_x],
            y=[min_dDGI],
            mode='markers',
            name='Min dDGI',
            marker=dict(color='red', size=7),
        ))

    max_dDGI = round(y_dgi.max()+4,-1)
    max_x = x_dgi.max()  # Get maximum x value
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'dDGI <i>vs</i> {xidx}',
            'font': dict(size=22, family="Arial Black", color="black"),
        },
        xaxis_title=xidx + unit,
        yaxis=dict(
            title='DGI (mm)',
            tickmode='linear',
            side='left',
            dtick=max_dDGI / 4,
            range=[0, max_dDGI * 1.05],
            showgrid=True,
            title_font=dict(size=18, family="Arial Black", color="black"),
            tickfont=dict(size=14, family="Arial Black", color="black")
        ),
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=dtick,
            showgrid=True,
            title_font=dict(size=18, family="Arial Black", color="black"),
            tickfont=dict(size=14, family="Arial Black", color="black"),
            range=[0, max_x * 1.05]
        ),
        font=dict(family="Arial Black", size=18, color="black"),
        legend=dict(
            x=1.08,
            y=1,
            xanchor='left',
            yanchor='top',
            font=dict(size=12),
            bgcolor='rgba(255, 255, 255, 0.7)',
            bordercolor='rgba(0, 0, 0, 0.3)',
            borderwidth=1
        ),
        showlegend=True
    )
    
    return fig, max_x, min_dDGI_x  # Return min_dDGI_x along with other values

def create_cdgi_plot(dgi_parameters_new, xidx, dtick, unit, prescript, max_x=None, min_dDGI_x=None):
    """Create cDGI plot using min_dDGI_x from dDGI plot"""
    fig = go.Figure()
    
    # Filter out any infinite or NaN values and reset index
    mask = np.isfinite(dgi_parameters_new[xidx]) & np.isfinite(dgi_parameters_new["cDGI"])
    x_data = dgi_parameters_new[xidx][mask].reset_index(drop=True)
    y_data = dgi_parameters_new["cDGI"][mask].reset_index(drop=True)
    
    # Add cDGI scatter plot
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='markers',
        marker=dict(color='royalblue'),
        name='cDGI',
        yaxis='y2'
    ))
    
    # Add regression line only if we have valid data
    if len(x_data) > 0 and len(y_data) > 0:
        try:
            x_array = x_data.to_numpy()
            y_array = y_data.to_numpy()
            xout, yout, wout = loess_1d(x_array, y_array, frac=.2)
            fig.add_trace(go.Scatter(
                x=xout,
                y=yout,
                mode='lines',
                marker=dict(color='lightskyblue'),
                name='Regression',
                yaxis='y2'
            ))
        except Exception as e:
            st.warning(f"Could not compute regression: {str(e)}")

    max_cdgi = round(y_data.max()+4,-1) if len(y_data) > 0 else 100
    if max_x is None:
        max_x = x_data.max() if len(x_data) > 0 else 100

    # Add minimum dDGI point using x-value from dDGI plot
    if min_dDGI_x is not None:
        closest_idx = (dgi_parameters_new[xidx] - min_dDGI_x).abs().idxmin()
        min_point_cDGI = dgi_parameters_new['cDGI'].loc[closest_idx]
        
        fig.add_trace(go.Scatter(
            x=[min_dDGI_x],
            y=[min_point_cDGI],
            mode='markers',
            name='Min dDGI',
            marker=dict(color='red', size=7),
            yaxis='y2'
        ))

    # Update layout
    fig.update_layout(
        title={
            'text': f'cDGI <i>vs</i> {xidx}',
            'font': dict(size=22, family="Arial Black", color="black"),
        },
        xaxis_title=xidx + unit,
        yaxis=dict(
            title='Relative Volume (%)',
            tickmode='linear',
            side='right',
            dtick=25,
            range=[0, 100 * 1.05],
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            title_font=dict(size=18, family="Arial Black", color="black"),
            tickfont=dict(size=14, family="Arial Black", color="black")
        ),
        yaxis2=dict(
            title='DGI (mm)',
            side='left',
            overlaying='y',
            tickmode='linear',
            dtick=max_cdgi / 4,
            range=[0, max_cdgi * 1.05],
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            title_font=dict(size=18, family="Arial Black", color="black"),
            tickfont=dict(size=14, family="Arial Black", color="black")
        ),
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=dtick,
            showgrid=True,
            title_font=dict(size=18, family="Arial Black", color="black"),
            tickfont=dict(size=14, family="Arial Black", color="black"),
            range=[0, max_x * 1.05]
        ),
        font=dict(family="Arial Black", size=18, color="black"),
        legend=dict(
            x=1.08,
            y=1,
            xanchor='left',
            yanchor='top',
            font=dict(size=12),
            bgcolor='rgba(255, 255, 255, 0.7)',
            bordercolor='rgba(0, 0, 0, 0.3)',
            borderwidth=1
        ),
        showlegend=True
    )
    
    return fig

def main():
    setup_page_style()
    
    # Initialize session state for processed data if it doesn't exist
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    # Get user inputs including show_isodose toggle
    uploaded_file, uploaded_structure_file, prescript_dose, min_dose, step_type, step_size, show_isodose = get_user_inputs()
    
    # Process DICOM files
    dicom_file, rtdose, rtss, RTstructures, structure_name_to_id = process_dicom_files(uploaded_file, uploaded_structure_file)
    
    # Get structure selections if available
    selected_structure_names = []
    if structure_name_to_id:
        selected_structure_names = st.sidebar.multiselect(
            "Select Structures for DVH Calculation", 
            list(structure_name_to_id.keys())
        )
    
    # Setup plot parameters
    unit, xidx, prescript, dunit, dtick = setup_plot_parameters(step_type, prescript_dose)
    
    if step_type == 'Relative (%)':
        step_size = round(prescript_dose * step_size * 0.01, 3)
    
    if st.sidebar.button('Process') or st.session_state.processed_data is not None:
        try:
            if st.session_state.processed_data is None:
                # Process dose data
                dose_data, ipp, pixel_spacing, grid_frame_offset_vector = read_dose_dicom(dicom_file)
                
                # Store processed data in session state
                st.session_state.processed_data = {
                    'dose_data': dose_data,
                    'ipp': ipp,
                    'pixel_spacing': pixel_spacing,
                    'grid_frame_offset_vector': grid_frame_offset_vector,
                    'dgi_parameters': None,
                    'dose_coordinates': None
                }
            else:
                # Use stored data
                dose_data = st.session_state.processed_data['dose_data']
                ipp = st.session_state.processed_data['ipp']
                pixel_spacing = st.session_state.processed_data['pixel_spacing']
                grid_frame_offset_vector = st.session_state.processed_data['grid_frame_offset_vector']
            
            # Validate prescription dose
            min_dose_value = np.min(dose_data)
            max_dose_value = np.max(dose_data)
            
            if prescript_dose < min_dose_value or prescript_dose > max_dose_value:
                st.error(f"Prescription dose should be between {min_dose_value} and {max_dose_value}.")
                return
            
            if st.session_state.processed_data['dgi_parameters'] is None:
                # Calculate DGI parameters
                dose_coordinates = extract_dose_coordinates_parallel(
                    dose_data, ipp, pixel_spacing, grid_frame_offset_vector, 
                    prescript_dose, min_dose, step_size, step_type
                )
                dgi_parameters = calculate_dgi(
                    dose_coordinates, min_dose, prescript_dose, step_size, step_type
                )
                
                # Store calculated parameters
                st.session_state.processed_data['dgi_parameters'] = dgi_parameters
                st.session_state.processed_data['dose_coordinates'] = dose_coordinates
            else:
                # Use stored parameters
                dgi_parameters = st.session_state.processed_data['dgi_parameters']
                dose_coordinates = st.session_state.processed_data['dose_coordinates']
            
            # Add download link
            st.sidebar.markdown(get_table_download_link(dgi_parameters), unsafe_allow_html=True)
            
            # Create and display plots with show_isodose parameter
            fig_ddgi, fig_cdgi = create_dgi_plots(
                dgi_parameters, xidx, prescript, dtick, unit, dunit,
                dose_data, ipp, pixel_spacing, grid_frame_offset_vector,
                step_size, min_dose, show_isodose,
                rtss, rtdose, RTstructures, structure_name_to_id, selected_structure_names
            )
            
            st.plotly_chart(fig_ddgi)
            st.plotly_chart(fig_cdgi)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.processed_data = None  # Clear stored data on error

    # Add footer at the end
    st.markdown("""
        <div class="footer">
            © 2024 Wonyoung Cho. All rights reserved. | Contact: 
            <a href="mailto:wycho@oncosoft.io">wycho@oncosoft.io</a>
        </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
