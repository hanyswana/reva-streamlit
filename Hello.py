import pandas as pd
import streamlit as st
import joblib
import requests

# Streamlit UI elements
st.title('Haemoglobin :')

def json_data():
    # First API call
    api_url1 = "https://x8ki-letl-twmt.n7.xano.io/api:3iQkTr3r/backgroundData"
    payload1 = {}
    response1 = requests.get(api_url1, params=payload1)

    if response1.status_code == 200:
        data1 = response1.json()
    else:
        st.write("Error in first API call:", response1.status_code)
        return None

    # Second API call
    api_url2 = "https://x8ki-letl-twmt.n7.xano.io/api:gTEeTJrZ/split_text"
    payload2 = {}
    response2 = requests.get(api_url2, params=payload2)

    if response2.status_code == 200:
        data2 = response2.json()
    else:
        st.write("Error in second API call:", response2.status_code)
        return None

    # Extract first line of data from both API responses and convert to numeric
    df1 = pd.DataFrame(data1).iloc[1:2].apply(pd.to_numeric, errors='coerce')
    df2 = pd.DataFrame(data2).iloc[:1].apply(pd.to_numeric, errors='coerce')

    st.write("Background:")
    st.write(df1)
    st.write("Sample:")
    st.write(df2)

    # Element-wise division of the dataframes
    absorbance_df = df1.div(df2.values).pow(2)

    absorbance_df.to_csv('absorbance_data.csv', index=False)
    return absorbance_df

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

# Load the new absorbance data 
absorbance_data = pd.read_csv('absorbance_data.csv')
st.write('Absorbance data:')
st.write(absorbance_data)

# Load the original data (124 samples)
ori_data = pd.read_csv('reva-lablink-oridata-124-x.csv')

# Combine the absorbance data with the original data
combined_data = pd.concat([absorbance_data, ori_data])
st.write('Combined Data:')
st.write(combined_data)

# Apply dimension reduction to the sample using Isomap and LLC
sample_iso = load_model('pipeline  104.csv_iso.joblib').fit_transform(combined_data)
sample_llc = load_model('pipeline  50.csv_llc.joblib').fit_transform(combined_data)

if len(combined_data) > 0:
    lr_prediction = lr_model.predict(combined_data)
    dtr_prediction = dtr_model.predict(combined_data)
    lr_iso_prediction = lr_iso_model.predict(sample_iso)
    dtr_iso_prediction = dtr_iso_model.predict(sample_iso)
    lr_llc_prediction = lr_llc_model.predict(sample_llc)
    dtr_llc_prediction = dtr_llc_model.predict(sample_llc)

    st.markdown(f"""
        <style>
            .hb_prediction {{
                font-size: 16px;
                font-weight: bold;
            }}
        </style>
        <p class="hb_prediction">(LR): {lr_prediction[0]} g/dL</p>
        <p class="hb_prediction">(DTR): {dtr_prediction[0]:.1f} g/dL</p>
        <p class="hb_prediction">(LR-ISOMAP): {lr_iso_prediction[0]:.1f} g/dL</p>
        <p class="hb_prediction">(DTR-ISOMAP): {dtr_iso_prediction[0]:.1f} g/dL</p>
        <p class="hb_prediction">(LR-LLC): {lr_llc_prediction[0]:.1f} g/dL</p>
        <p class="hb_prediction">(DTR-LLC): {dtr_llc_prediction[0]:.1f} g/dL</p>
    """, unsafe_allow_html=True)
else:
    st.write('The sample is empty. Please load a sample with data.')
