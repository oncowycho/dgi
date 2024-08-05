import streamlit as st
import pydicom
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from skimage.measure import find_contours
import plotly.graph_objects as go
import base64
from multiprocessing import Pool

st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def read_dose_dicom(dicom_file_path):
    ds = pydicom.dcmread(dicom_file_path)
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

def extract_dose_coordinates_parallel(dose_data, ipp, pixel_spacing, grid_frame_offset_vector, min_dose=1, step_size=1):
    vmin = max(round(np.min(dose_data), -1), 1)
    vmax = np.max(dose_data)
    dose_values = np.arange(vmin, vmax, step_size)
    with Pool() as pool:
        results = pool.map(process_slice, [(z, dose_data[z], ipp, pixel_spacing, grid_frame_offset_vector, dose_values) for z in range(dose_data.shape[0])])
    coordinates = [item for sublist in results for item in sublist]
    return coordinates

def extract_dose_coordinates(dose_data, ipp, pixel_spacing, grid_frame_offset_vector, min_dose=1, step_size=1):
    mask = dose_data >= min_dose
    coords = np.argwhere(mask)
    doses = dose_data[mask]

    vmin = max(round(np.min(dose_data), -1), 1)
    vmax = np.max(dose_data)
    
    dose_values = np.arange(vmin, vmax, step_size)

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
    
def calculate_dgi(dose_coordinates, min_dose, prescript_dose):
    df = pd.DataFrame(dose_coordinates, columns=["x", "y", "z", "dose"]).sort_values(by='dose')
    df = df[df.dose >= min_dose]
    df = df.drop_duplicates()

    df['dose'] = df['dose'].astype('category')
    grouped = df['dose'].cat.categories

    dgi_para = []
    area0 = 0
    volume0 = 0
    cDGI = None
    
    for dose in grouped[::-1]:
        points = df[df['dose'] == dose][['x', 'y', 'z']].values

        if len(points) >= 4: 
            hull = ConvexHull(points, qhull_options='QJ')
            area, volume = hull.area, hull.volume
        else:
            continue
        
        dDGI = (volume - volume0) / (0.5 * (area + area0))
        pdose = (dose / prescript_dose) * 100
        
        if dose >= min_dose and dose != grouped[-1]:
            if dose < prescript_dose:    
                cDGI += dDGI
            elif dose == prescript_dose: 
                cDGI = 0

            dgi_para.append([dose, pdose, area, volume, dDGI, cDGI])

        if dose != grouped[-1]: 
            area0, volume0 = area, volume
            
    return pd.DataFrame(dgi_para, columns=["Dose", "Dose(%)","Area", "Volume", "dDGI", "cDGI"])

def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="dgi_parameters.csv">Download dgi_parameters.csv</a>'
    return href

# Streamlit UI
st.title('Dose Gradient Curve')

# Inputs
uploaded_file = st.file_uploader("Choose a DICOM file (optional)", type=["dcm"])
prescript_dose = st.number_input('Prescription Dose', min_value=0, value=70)
min_dose = st.number_input('Minimum Dose', min_value=0, value=1)
step_type = st.radio('Dose step size',['Absolute', 'Relative'], horizontal=True)
step_size = st.number_input('Step Size', min_value=0.5, max_value=1.5, value=1.0, step=0.5, format="%.2f",label_visibility="collapsed")


# Use the default file if no file is uploaded
dicom_path = 'dose.dcm'
if uploaded_file is not None:
    dicom_path = uploaded_file.name
    with open(dicom_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

if st.button('Process'):
    try:
        dose_data, ipp, pixel_spacing, grid_frame_offset_vector = read_dose_dicom(dicom_path)
        min_dose_value = np.min(dose_data)
        max_dose_value = np.max(dose_data)

        if prescript_dose < min_dose_value or prescript_dose > max_dose_value:
            st.error(f"Prescription dose should be between {min_dose_value} and {max_dose_value}.")
        else:
            dose_coordinates = extract_dose_coordinates(dose_data, ipp, pixel_spacing, grid_frame_offset_vector, min_dose, step_size)
            dgi_parameters = calculate_dgi(dose_coordinates, min_dose, prescript_dose)

            # Save CSV file
            st.markdown(get_table_download_link(dgi_parameters), unsafe_allow_html=True)

            # Plot dDGI
            fig_ddgi = go.Figure()
            if step_type == 'Absolute':
                fig_ddgi.add_trace(go.Scatter(x=dgi_parameters["Dose"], y=dgi_parameters["dDGI"], mode='markers', name='dDGI'))
                fig_ddgi.update_layout(title='dDGI vs Dose', xaxis_title='Dose', yaxis_title='dDGI')
            elif step_type == 'Relative':
                fig_ddgi.add_trace(go.Scatter(x=dgi_parameters["Dose(%)"], y=dgi_parameters["dDGI"], mode='markers', name='dDGI'))
                fig_ddgi.update_layout(title='dDGI vs Dose (%)', xaxis_title='Dose (%)', yaxis_title='dDGI')

            st.plotly_chart(fig_ddgi)

            # Plot cDGI
            fig_cdgi = go.Figure()
            fig_cdgi.add_trace(go.Scatter(x=dgi_parameters["Dose(%)"], y=dgi_parameters["cDGI"], mode='markers', name='cDGI'))
            fig_cdgi.update_layout(title='cDGI vs Dose (%)', xaxis_title='Dose (%)', yaxis_title='cDGI')
            st.plotly_chart(fig_cdgi)
    except Exception as e:
        st.error(f"An error occurred: {e}")
