import pandas as pd
import streamlit as st
import joblib
from flask import request
# from streamlit.web.server.server import Server

# # Enable CORS
# Server.enableCORS = True


# # @middleware
# def enable_cors(response):
#     response.headers["Access-Control-Allow-Origin"] = "*"
#     return response


# def receive_data(sendData):
#     data = request.get_json(sendData) 
#     st.write(data)

# ff_data = receive_data()
# print('Json data')
# print(ff_data)

def receive_data():
    data = st.session_state['sendData']
    st.write(data)

if 'sendData' not in st.session_state:
    st.session_state['sendData'] = None

data_received = receive_data()

if data_received:
    st.write('Data received:', data_received)

# Load a model from the pickle file
def load_model(model_file):
    with open(model_file, 'rb') as f:
        model = joblib.load(f)
    return model


# Streamlit UI elements
st.title('REVA (Hb Prediction)')

# Load the ori data (124 samples)
ori_data = pd.read_csv('reva-lablink-oridata-40.csv')

# Load the new data (1 sample) and convert json to csv file
json_data = pd.read_json('example1.json')
json_data.to_csv('example1.csv', index=False)
new_data = pd.read_csv('example1.csv')

# Combine the ori data with the new data
sample_data = pd.concat([new_data, ori_data])

# st.write('Spectral Data:')
# st.write(example)
# st.write(merged_data)

lr_model = load_model('pipeline  6.csv_lr_ori.joblib')
dtr_model = load_model('pipeline  64.csv_dtr_ori.joblib')
lr_iso_model = load_model('pipeline  85.csv_lr_iso.joblib')
dtr_iso_model = load_model('pipeline  63.csv_dtr_iso.joblib')
lr_llc_model = load_model('pipeline  78.csv_lr_llc.joblib')
dtr_llc_model = load_model('pipeline  92.csv_dtr_llc.joblib')

# Apply dimension reduction to the sample using Isomap and LLC
sample_iso = load_model('pipeline  104.csv_iso.joblib').fit_transform(sample_data)
sample_llc = load_model('pipeline  50.csv_llc.joblib').fit_transform(sample_data)

# Streamlit UI elements
st.write('Click the "Predict" button to make prediction of Hb.')

# Add button to trigger prediction
if st.button('Predict'):
    if len(sample_data) > 0:
        lr_prediction = lr_model.predict(sample_data)
        dtr_prediction = dtr_model.predict(sample_data)
        lr_iso_prediction = lr_iso_model.predict(sample_iso)
        dtr_iso_prediction = dtr_iso_model.predict(sample_iso)
        lr_llc_prediction = lr_llc_model.predict(sample_llc)
        dtr_llc_prediction = dtr_llc_model.predict(sample_llc)

        st.markdown(f"""
            <style>
                .hb_prediction {{
                    font-size: 18px;
                    font-weight: bold;
                }}
            </style>
            <p class="hb_prediction">Hb value (LR): {lr_prediction[0]} g/dL</p>
            <p class="hb_prediction">Hb value (DTR): {dtr_prediction[0]:.1f} g/dL</p>
            <p class="hb_prediction">Hb value (LR-ISOMAP): {lr_iso_prediction[0]:.1f} g/dL</p>
            <p class="hb_prediction">Hb value (DTR-ISOMAP): {dtr_iso_prediction[0]:.1f} g/dL</p>
            <p class="hb_prediction">Hb value (LR-LLC): {lr_llc_prediction[0]:.1f} g/dL</p>
            <p class="hb_prediction">Hb value (DTR-LLC): {dtr_llc_prediction[0]:.1f} g/dL</p>
        """, unsafe_allow_html=True)
    else:
        st.write('The sample is empty. Please load a sample with data.')
