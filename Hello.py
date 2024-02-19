import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt
import tensorflow as tf

st.markdown("""
<style>
.custom-font {font-size: 18px; font-weight: bold;}
</style> """, unsafe_allow_html=True)

st.markdown('<p class="custom-font">Absorbance data :</p>', unsafe_allow_html=True)

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
    df1 = pd.DataFrame(data1).iloc[:1].apply(pd.to_numeric, errors='coerce')
    df2 = pd.DataFrame(data2).iloc[:1].apply(pd.to_numeric, errors='coerce')
    wavelengths = df1.columns

    # Element-wise division of the dataframes & convert absorbance data to csv
    absorbance_df = df1.div(df2.values).pow(2)
    st.write(absorbance_df)

    # Convert DataFrame to CSV
    csv_data = absorbance_df.to_csv(index=False)
    
    # Create a download button and offer the CSV to download
    st.download_button(
        label="Download Absorbance Data as CSV",
        data=csv_data,
        file_name="absorbance_data.csv",
        mime="text/csv",
    )
    
    absorbance_data = absorbance_df.iloc[0]  # First row of absorbance data

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, absorbance_data, marker='o', linestyle='-', color='b')
    plt.xlabel('Wavelength (nm)', fontweight='bold', fontsize=14)
    plt.ylabel('Absorbance', fontweight='bold', fontsize=14)
    plt.xticks(rotation='vertical', fontweight='bold', fontsize=12)
    plt.yticks(fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)
 
    return absorbance_df

json_data()

def load_model(model_dir):
    model = tf.saved_model.load(model_dir)
    return model

def predict_with_model(model, input_data):

    # Convert DataFrame to numpy array with dtype 'float64' to match model's expectation
    input_array = input_data.to_numpy(dtype='float64')
    
    # Ensure the input_array has the correct shape, (-1, 19), where -1 is any batch size
    # and 19 is the number of features
    input_array_reshaped = input_array.reshape(-1, 19)  # Adjust to match the number of features your model expects
    
    # Convert reshaped array to tensor with dtype=tf.float64
    input_tensor = tf.convert_to_tensor(input_array_reshaped, dtype=tf.float64)
    
    # Use the model for prediction
    # Assuming the model has a predict function, which is common for TensorFlow models
    predictions = model(input_tensor)
    
    return predictions.numpy()  # Convert predictions to numpy array if needed

def main():
    # Load the TensorFlow model
    model_path = 'reva-lablink-hb-125-(original-data)-15-02-24'
    model = load_model(model_path)

    # Get data from server (simulated here)
    absorbance_data = json_data()

    # Predict
    predictions = predict_with_model(model, absorbance_data)
    predictions_value = predictions[0][0]

    st.markdown("""
    <style>
    .custom-font {font-size: 18px; -weight: bold;}
    .high-value {color: red;}
    </style> """, unsafe_allow_html=True)
    
        # Add condition for prediction value
    if predictions_value > 25:
        display_value = f'<span class="high-value">High value : ({predictions_value:.1f} g/dL)</span>'
    else:
        display_value = f"{predictions_value:.1f} g/dL"
    
    st.markdown(f'<p class="custom-font">Haemoglobin :<br>{display_value}</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
