import pandas as pd
import streamlit as st
import joblib


# Load a model from the pickle file
def load_model(model_file):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model


# Streamlit UI elements
st.title('REVA - Haemoglobin (Hb) Prediction')

# Load the sample
sample = pd.read_csv('reva-lablink-oridata-20.csv')
# sample = pickle.load(open('reva-lablink-oridata.pkl', 'rb'))
# st.write('Spectral Data:')
# st.write(sample)

lr_model = load_model('pipeline  6.csv_lr_ori.joblib')
dtr_model = load_model('pipeline  64.csv_dtr_ori.joblib')
lr_iso_model = load_model('pipeline  85.csv_lr_iso.joblib')
dtr_iso_model = load_model('pipeline  63.csv_dtr_iso.joblib')
lr_llc_model = load_model('pipeline  78.csv_lr_llc.joblib')
dtr_llc_model = load_model('pipeline  92.csv_dtr_llc.joblib')

# Apply dimension reduction to the sample using Isomap and LLC
sample_iso = load_model('pipeline  104.csv_iso.joblib').fit_transform(sample)
sample_llc = load_model('pipeline  50.csv_llc.joblib').fit_transform(sample)

# Streamlit UI elements
st.write('Click the "Predict" button to make prediction of Hb.')

# Add button to trigger prediction
if st.button('Predict'):
    if len(sample) > 0:
        lr_prediction = lr_model.predict(sample)
        dtr_prediction = dtr_model.predict(sample)
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
            <p class="hb_prediction">Hb prediction (LR): {lr_prediction[0]} g/dl</p>
            <p class="hb_prediction">Hb prediction (DTR): {dtr_prediction[0]:.1f} g/dl</p>
            <p class="hb_prediction">Hb prediction (LR-ISOMAP): {lr_iso_prediction[0]:.1f} g/dl</p>
            <p class="hb_prediction">Hb prediction (DTR-ISOMAP): {dtr_iso_prediction[0]:.1f} g/dl</p>
            <p class="hb_prediction">Hb prediction (LR-LLC): {lr_llc_prediction[0]:.1f} g/dl</p>
            <p class="hb_prediction">Hb prediction (DTR-LLC): {dtr_llc_prediction[0]:.1f} g/dl</p>
        """, unsafe_allow_html=True)
    else:
        st.write('The sample is empty. Please load a sample with data.')
