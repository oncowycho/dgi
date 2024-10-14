import streamlit as st
import pydicom
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from skimage.measure import find_contours
import plotly.graph_objects as go
import base64
from loess.loess_1d import loess_1d
from multiprocessing import Pool
from pydicom.uid import ImplicitVRLittleEndian


st.set_page_config(layout="wide")

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

# ------------------------ [ UI ] -----------------------------
st.title('Dose Gradient Curve')

# Inputs
st.sidebar.header("Upload DICOM Files")
uploaded_file = st.sidebar.file_uploader("Upload RT Dose (dose.dcm)", type=["dcm"])
structure_file = st.sidebar.file_uploader("Upload RT Structure (rts.dcm) (Optional)", type=["dcm"])

prescript_dose = st.sidebar.number_input('Prescription Dose (Gy)', min_value=0.0, value=40.0, format="%.2f")
min_dose = st.sidebar.number_input('Minimum Dose (Gy)', min_value=0.1, value=1.0, step=0.1, format="%.2f")
step_type = st.sidebar.radio('Dose step size',['Absolute (Gy)', 'Relative (%)'], horizontal=True)
unit = ' (Gy)'
if step_type == 'Relative (%)': unit = ''
step = 0.1; fmt = '%.2f'
# if step_type == 'Absolute (Gy)': step = 0.01; fmt = '%.2f'
step_size = round(st.sidebar.number_input('Step Size', min_value=step, max_value=9.0, value=1.0, step=step, format=fmt,label_visibility="collapsed"),3)

if step_type == 'Relative (%)': step_size = round(prescript_dose*step_size*0.01,3)

# Use the default file if no file is uploaded
dicom_path = 'dose.dcm'
if uploaded_file is not None:
    dicom_path = uploaded_file.name
    with open(dicom_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

if structure_file:
    rtss_file = pydicom.dcmread(structure_file, force=True)
    rtss_parser = dicomparser.DicomParser(rtss_file)

    RTstructures = rtss_parser.GetStructures()

    # Create a dictionary mapping structure names to structure IDs
    structure_name_to_id = {structure['name']: key for key, structure in RTstructures.items()}

    # Using multiselect for selecting multiple structures by name
    selected_structure_names = st.sidebar.multiselect(
        "Select Structures for DVH Calculation", list(structure_name_to_id.keys())
    )
    
if st.sidebar.button('Process'):
    try:
        dose_data, ipp, pixel_spacing, grid_frame_offset_vector = read_dose_dicom(dicom_path)
        min_dose_value = np.min(dose_data)
        max_dose_value = np.max(dose_data)

        if prescript_dose < min_dose_value or prescript_dose > max_dose_value:
            st.error(f"Prescription dose should be between {min_dose_value} and {max_dose_value}.")
        else:
            dose_coordinates = extract_dose_coordinates(dose_data, ipp, pixel_spacing, grid_frame_offset_vector, prescript_dose, min_dose, step_size, step_type)
            dgi_parameters = calculate_dgi(dose_coordinates, min_dose, prescript_dose, step_size, step_type)

            # Save CSV file
            st.sidebar.markdown(get_table_download_link(dgi_parameters), unsafe_allow_html=True)

            if step_type == 'Absolute (Gy)':
                xidx = 'Dose'
                prescript = prescript_dose
            elif step_type == 'Relative (%)':
                xidx = 'Dose (%)'
                prescript = 100
                
            # Plot dDGI
            fig_ddgi = go.Figure()
            fig_ddgi.add_trace(go.Scatter(x=dgi_parameters[xidx], y=dgi_parameters["dDGI"], mode='markers', name='dDGI'))

            xout, yout, wout = loess_1d(dgi_parameters[xidx].values, dgi_parameters["dDGI"].values, frac=.2)
            fig_ddgi.add_trace(go.Scatter(x=xout, y=yout, mode='lines', name='Regression'))

            # Find the minimum dDGI near the prescription dose 10% range around the prescription dose
            range_around_prescript = 0.1 * prescript
            nearby_points = dgi_parameters[(dgi_parameters[xidx] >= prescript - range_around_prescript) & 
                                           (dgi_parameters[xidx] <= prescript + range_around_prescript)]
            if not nearby_points.empty:
                min_dDGI = nearby_points["dDGI"].min()
                min_dDGI_idx = nearby_points["dDGI"].idxmin()
                min_dDGI_point = nearby_points[xidx].loc[min_dDGI_idx]
                fig_ddgi.add_trace(go.Scatter(x=[min_dDGI_point], y=[min_dDGI], mode='markers+text', name='Min dDGI',
                                              marker=dict(color='red'), text=["Min dDGI"], textposition="top center"))

            fig_ddgi.update_layout(title=f'dDGI vs {xidx}', xaxis_title=xidx+unit, yaxis_title='DGI (mm)')
            
            st.plotly_chart(fig_ddgi)

            # Plot cDGI
            dgi_parameters = dgi_parameters.dropna()
            fig_cdgi = go.Figure()
            fig_cdgi.add_trace(go.Scatter(x=dgi_parameters[xidx], y=dgi_parameters["cDGI"], mode='markers', name='cDGI'))
            fig_cdgi.update_layout(title=f'cDGI vs {xidx}', xaxis_title=xidx+unit, yaxis_title='DGI (mm)')

            xout, yout, wout = loess_1d(dgi_parameters[xidx].values, dgi_parameters["cDGI"].values, frac=.2)
            fig_cdgi.add_trace(go.Scatter(x=xout, y=yout, mode='lines', name='Regression'))

            min_dDGI_on_cDGI = nearby_points['cDGI'].loc[min_dDGI_idx]
            fig_cdgi.add_trace(go.Scatter(x=[min_dDGI_point], y=[min_dDGI_on_cDGI], mode='markers+text', name='Min dDGI',
                                          marker=dict(color='red'), text=["Min dDGI"], textposition="top center"))
            
            st.plotly_chart(fig_cdgi)
    except Exception as e:
        st.error(f"An error occurred: {e}")
