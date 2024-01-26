import pandas as pd
import streamlit as st
import joblib
import requests


def json_data():
    
    # Define the API URL from server (Xano) and payload
    api_url = "https://x8ki-letl-twmt.n7.xano.io/api:gTEeTJrZ/split_text"
    payload = {}

    # Make the API request and store the response
    response = requests.get(api_url, params=payload)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON data
        data = response.json()

        # Convert json to csv
        df = pd.DataFrame(data)
        df.iloc[:1].to_csv('json_data.csv', index=False)
        return df.iloc[:1]
    else:
        # Display an error message
        st.write("Error:", response.status_code)
        return None


def main():
    # Get data from server (Xano)
    data_df = json_data()


if __name__ == "__main__":
    main()


# Load a model from the pickle file
def load_model(model_file):
    with open(model_file, 'rb') as f:
        model = joblib.load(f)
    return model
    
lr_model = load_model('pipeline  6.csv_lr_ori.joblib')
dtr_model = load_model('pipeline  64.csv_dtr_ori2.joblib')
lr_iso_model = load_model('pipeline  85.csv_lr_iso.joblib')
dtr_iso_model = load_model('pipeline  63.csv_dtr_iso2.joblib')
lr_llc_model = load_model('pipeline  78.csv_lr_llc.joblib')
dtr_llc_model = load_model('pipeline  92.csv_dtr_llc2.joblib')


# Streamlit UI elements
st.title('Prediction')

# Load the new data (1 sample) 
new_data = pd.read_csv('json_data.csv')
st.write('Spectral data:')
st.write(new_data)

# Load the ori data (124 samples)
ori_data = pd.read_csv('reva-lablink-oridata-124-x.csv')
# st.write('Ori Data:')
# st.write(ori_data)

# Combine the ori data with the new data
sample_data = pd.concat([new_data.iloc[:1], ori_data])
# st.write('Sample Data:')
# st.write(sample_data)

# Apply dimension reduction to the sample using Isomap and LLC
sample_iso = load_model('pipeline  104.csv_iso.joblib').fit_transform(sample_data)
sample_llc = load_model('pipeline  50.csv_llc.joblib').fit_transform(sample_data)


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


# # Streamlit UI elements
# st.write('Click the "Predict" button to make prediction of Hb.')

# # Add button to trigger prediction
# if st.button('Predict'):
#     if len(sample_data) > 0:
#         lr_prediction = lr_model.predict(sample_data)
#         dtr_prediction = dtr_model.predict(sample_data)
#         lr_iso_prediction = lr_iso_model.predict(sample_iso)
#         dtr_iso_prediction = dtr_iso_model.predict(sample_iso)
#         lr_llc_prediction = lr_llc_model.predict(sample_llc)
#         dtr_llc_prediction = dtr_llc_model.predict(sample_llc)

#         st.markdown(f"""
#             <style>
#                 .hb_prediction {{
#                     font-size: 18px;
#                     font-weight: bold;
#                 }}
#             </style>
#             <p class="hb_prediction">Hb value (LR): {lr_prediction[0]} g/dL</p>
#             <p class="hb_prediction">Hb value (DTR): {dtr_prediction[0]:.1f} g/dL</p>
#             <p class="hb_prediction">Hb value (LR-ISOMAP): {lr_iso_prediction[0]:.1f} g/dL</p>
#             <p class="hb_prediction">Hb value (DTR-ISOMAP): {dtr_iso_prediction[0]:.1f} g/dL</p>
#             <p class="hb_prediction">Hb value (LR-LLC): {lr_llc_prediction[0]:.1f} g/dL</p>
#             <p class="hb_prediction">Hb value (DTR-LLC): {dtr_llc_prediction[0]:.1f} g/dL</p>
#         """, unsafe_allow_html=True)
#     else:
#         st.write('The sample is empty. Please load a sample with data.')







