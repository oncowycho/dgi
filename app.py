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

st.set_page_config(layout="wide",page_title = "Dose Gradient Curve", page_icon="logo_tab.ico",)
theme = st.get_option("theme.base")

if theme == "dark":
    st.logo('logo_light.png')
else:
    st.logo('logo_dark.png')

#st.logo('logo.png')

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


def cal_dvh(fig, rtss, rtdose, RTstructures, structure_name_to_id, selected_structure_names, unit):
    calcdvhs = {}

    # Loop through the selected structure names and get the corresponding structure IDs
    for structure_name in selected_structure_names:
        structure_id = structure_name_to_id[structure_name]
        structure = RTstructures[structure_id]
        calcdvhs[structure_id] = get_dvh(rtss, rtdose, structure_id)

        if calcdvhs[structure_id].counts.any():
            # Add the DVH plot for each selected structure
            fig.add_trace(go.Scatter(
                x=np.arange(0,len(calcdvhs[structure_id].counts))/unit,
                y=calcdvhs[structure_id].counts * 100 / calcdvhs[structure_id].counts[0],
                mode='lines',
                name=structure['name'],
                line=dict(color=f'rgb({structure["color"][0]}, {structure["color"][1]}, {structure["color"][2]})', dash='dash'),
                yaxis='y'
            ))

    return fig
    

def main():
    # ------------------------ [ UI ] -----------------------------
    st.title('Dose Gradient Curve')

    # Inputs
    st.sidebar.header("Upload DICOM Files")
    uploaded_file = st.sidebar.file_uploader("Upload RT Dose (dose.dcm)", type=["dcm"])
    structure_file = st.sidebar.file_uploader("Upload RT Structure (rts.dcm) (Optional)", type=["dcm"])
    
    prescript_dose = st.sidebar.number_input('Prescription Dose (Gy)', min_value=0.0, value=40.0, format="%.2f")
    min_dose = st.sidebar.number_input('Minimum Dose (Gy)', min_value=0.1, value=0.1, step=0.1, format="%.2f")
    step_type = st.sidebar.radio('Dose step size',['Absolute (Gy)', 'Relative (%)'], horizontal=True)
    unit = ' (Gy)'
    step = 0.1; fmt = '%.2f'
    
    # if step_type == 'Absolute (Gy)': step = 0.01; fmt = '%.2f'
    step_size = round(st.sidebar.number_input('Step Size', min_value=step, max_value=9.0, value=1.0, step=step, format=fmt,label_visibility="collapsed"),3)
    
    if step_type == 'Relative (%)':
        step_size = round(prescript_dose*step_size*0.01,3)
        unit = ''

    if step_type == 'Absolute (Gy)':
        xidx = 'Dose'
        prescript = prescript_dose
        dunit = 100
        dtick = 2
    elif step_type == 'Relative (%)':
        xidx = 'Dose (%)'
        prescript = 100
        dunit = prescript_dose
        dtick = 5

    # Use the default file if no file is uploaded
    dicom_file = 'dose.dcm'
    if uploaded_file is not None:
        dicom_file = uploaded_file.name
        with open(dicom_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
    if dicom_file:
        rtdose_file = pydicom.dcmread(dicom_file, force=True)
        if not hasattr(rtdose_file.file_meta, 'TransferSyntaxUID'):
            rtdose_file.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
        rtdose = dicomparser.DicomParser(rtdose_file)
          
    if structure_file:
        rtss_file = pydicom.dcmread(structure_file, force=True)

        # Check SOP Instance UID or Study Instance UID to ensure the files belong to the same study
        if hasattr(rtss_file, 'StudyInstanceUID') and hasattr(rtdose_file, 'StudyInstanceUID'):
            if rtss_file.StudyInstanceUID != rtdose_file.StudyInstanceUID:
                structure_file = None
            else:
                rtss = dicomparser.DicomParser(rtss_file)
                RTstructures = rtss.GetStructures()
        
                # Create a dictionary mapping structure names to structure IDs
                structure_name_to_id = {structure['name']: key for key, structure in RTstructures.items()}

                # Using multiselect for selecting multiple structures by name
                selected_structure_names = st.sidebar.multiselect(
                    "Select Structures for DVH Calculation", list(structure_name_to_id.keys())
                )
        else:
            structure_file = None
        
    if st.sidebar.button('Process'):
        fig_cdgi = go.Figure()
        if structure_file:
            fig_cdgi = cal_dvh(fig_cdgi, rtss, rtdose, RTstructures, structure_name_to_id, selected_structure_names, dunit)
        
        try:
            dose_data, ipp, pixel_spacing, grid_frame_offset_vector = read_dose_dicom(dicom_file)
            min_dose_value = np.min(dose_data)
            max_dose_value = np.max(dose_data)
            
            if prescript_dose < min_dose_value or prescript_dose > max_dose_value:
                st.error(f"Prescription dose should be between {min_dose_value} and {max_dose_value}.")
            else:
                dose_coordinates = extract_dose_coordinates_parallel(dose_data, ipp, pixel_spacing, grid_frame_offset_vector, prescript_dose, min_dose, step_size, step_type)
                dgi_parameters = calculate_dgi(dose_coordinates, min_dose, prescript_dose, step_size, step_type)

                # Save CSV file
                st.sidebar.markdown(get_table_download_link(dgi_parameters), unsafe_allow_html=True)

                # Plot dDGI
                fig_ddgi = go.Figure()
                fig_ddgi.add_trace(go.Scatter(x=dgi_parameters[xidx], y=dgi_parameters["dDGI"],
                                              mode='markers', marker=dict(color='royalblue'), name='dDGI')
                                   )

                xout, yout, wout = loess_1d(dgi_parameters[xidx].values, dgi_parameters["dDGI"].values, frac=.2)
                fig_ddgi.add_trace(go.Scatter(x=xout, y=yout, mode='lines', marker=dict(color='lightskyblue'), name='Regression'))
                
                # Find the minimum dDGI near the prescription dose 10% range around the prescription dose
                range_around_prescript = 0.1 * prescript
                nearby_points = dgi_parameters[(dgi_parameters[xidx] >= prescript - range_around_prescript) & 
                                               (dgi_parameters[xidx] <= prescript + range_around_prescript)]
                if not nearby_points.empty:
                    min_dDGI = nearby_points["dDGI"].min()
                    min_dDGI_idx = nearby_points["dDGI"].idxmin()
                    min_dDGI_point = nearby_points[xidx].loc[min_dDGI_idx]
                    fig_ddgi.add_trace(go.Scatter(x=[min_dDGI_point], y=[min_dDGI], mode='markers+text', name='Min dDGI',
                                                  marker=dict(color='red'),
                                                  text=["Min dDGI"], textposition="top center", textfont=dict(size=14, color='gray')))

                max_dDGI = round(dgi_parameters["dDGI"].max()+4,-1)
                
                fig_ddgi.update_layout(
                    title=f'dDGI vs {xidx}',
                    xaxis_title=xidx+unit,
                    yaxis=dict(
                        title='DGI (mm)',
                        tickmode='linear',
                        side='left',
                        dtick=max_dDGI/4,
                        range=[0, max_dDGI*1.05],
                        showgrid=True
                    ),
                    xaxis=dict(
                        tickmode='linear',
                        tick0=0,
                        dtick=dtick,
                        showgrid=True
                    ),

                    font=dict(family="Arial, bold", size=18, color="black"),
                    # height=400,
                    legend=dict(x=1.06, y=1,
                                xanchor='left', yanchor='top',
                                font=dict(size=12))
                )
            
                st.plotly_chart(fig_ddgi)

                # Plot cDGI
                dgi_parameters = dgi_parameters.dropna()
                # fig_cdgi = go.Figure()
                fig_cdgi.add_trace(go.Scatter(x=dgi_parameters[xidx], y=dgi_parameters["cDGI"],
                                              mode='markers', marker=dict(color='royalblue'),
                                              name='cDGI', yaxis='y2')
                                   )

                max_cdgi = round(dgi_parameters["cDGI"].max()+4,-1)

                fig_cdgi.update_layout(
                    title=f'cDGI vs {xidx}',
                    xaxis_title=xidx+unit,
                    yaxis=dict(
                        title='Relative Volume (%)',
                        tickmode='linear',
                        side='right',
                        dtick=25,
                        range=[0, 100*1.05],
                        showgrid=True
                    ),
                    yaxis2=dict(
                        title='DGI (mm)',
                        side='left',
                        overlaying='y',
                        # matches='y',
                        tickmode='linear',
                        dtick=max_cdgi/4,
                        range=[0, max_cdgi*1.05],
                        showgrid=False
                    ),
                    xaxis=dict(
                        tickmode='linear',
                        tick0=0,
                        dtick=dtick,
                        showgrid=True
                    ),
                    font=dict(family="Arial, bold", size=18, color="black"),
                    # height=600,
                    legend=dict(
                        x=1.06,
                        y=1,
                        xanchor='left',
                        yanchor='top',
                        font=dict(size=12)
                    )
                )
                
                xout, yout, wout = loess_1d(dgi_parameters[xidx].values, dgi_parameters["cDGI"].values, frac=.2)
                fig_cdgi.add_trace(go.Scatter(x=xout, y=yout, mode='lines', name='Regression', marker=dict(color='lightskyblue'),yaxis='y2'))
                
                min_dDGI_on_cDGI = nearby_points['cDGI'].loc[min_dDGI_idx]
                fig_cdgi.add_trace(go.Scatter(x=[min_dDGI_point], y=[min_dDGI_on_cDGI],
                                              mode='markers+text', name='Min dDGI',
                                              marker=dict(color='red'), text=["Min dDGI"], textposition="top center", textfont=dict(size=14, color="gray"),yaxis='y2')
                                   )
                
                st.plotly_chart(fig_cdgi)
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
