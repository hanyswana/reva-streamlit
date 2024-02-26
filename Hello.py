import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import Normalizer

# st.markdown("""
# <style>
# .custom-font {font-size: 16px; font-weight: bold;}
# </style> """, unsafe_allow_html=True)

# st.markdown('<p class="custom-font">Absorbance data :</p>', unsafe_allow_html=True)

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

    # Normalize the absorbance data using Euclidean normalization
    normalizer = Normalizer(norm='l2')  # Euclidean normalization
    absorbance_normalized_euc = normalizer.transform(absorbance_df)
    absorbance_normalized_euc_df = pd.DataFrame(absorbance_normalized_euc, columns=absorbance_df.columns)
    st.write(absorbance_normalized_euc_df)

    # Convert normalized DataFrame to CSV (optional step, depending on your needs)
    absorbance_normalized_euc_df.to_csv('absorbance_data_normalized_euc.csv', index=False)

    # # First row of absorbance data
    # absorbance_data = absorbance_normalized_df.iloc[0]  
 
    return absorbance_df, absorbance_normalized_euc_df, wavelengths

def load_model(model_dir):
    model = tf.saved_model.load(model_dir)
    return model

def predict_with_model(model, input_data):

    input_array = input_data.to_numpy(dtype='float64')
    input_array_reshaped = input_array.reshape(-1, 19)  # Adjust to match the number of features your model expects
    input_tensor = tf.convert_to_tensor(input_array_reshaped, dtype=tf.float64)
    predictions = model(input_tensor)
    return predictions.numpy()  # Convert predictions to numpy array if needed

def main():
    # Define model paths with labels
    model_paths_with_labels = [
        ('R39', 'reva-lablink-hb-125-(original-data).csv_r2_0.39_2024-02-15_11-55-27'),
        ('R26', 'reva-lablink-hb-125-(original-data).csv_best_model_2024-02-16_17-44-04_b4_r0.26')
    ]

    # Get data from server (simulated here)
    absorbance_data, absorbance_normalized_euc_data, wavelengths = json_data()

    for label, model_path in model_paths_with_labels:
        # Load the model
        model = load_model(model_path)
        # st.write(model)
        
        # Predict with original absorbance data
        predictions_original = predict_with_model(model, absorbance_data)
        predictions_value_original = predictions_original[0][0]
        
        # Predict with Euclidean normalized absorbance data
        predictions_normalized_euc = predict_with_model(model, absorbance_normalized_euc_data)
        predictions_value_normalized_euc = predictions_normalized_euc[0][0]
    
        st.markdown("""
        <style>
        .label {font-size: 16px; font-weight: bold; color: black;}
        .value {font-size: 30px; font-weight: bold; color: blue;}
        .high-value {color: red;}
        </style> """, unsafe_allow_html=True)
    
        # Add condition for prediction value
        if predictions_value_original > 25:
            display_value = f'<span class="high-value">High value : ({predictions_value_original:.1f} g/dL)</span>'
            display_value2 = f'<span class="high-value">High value : ({predictions_value_normalized_euc:.1f} g/dL)</span>'
        else:
            display_value = f'<span class="value">{predictions_value_original:.1f} g/dL</span>'
            display_value2 = f'<span class="value">{predictions_value_normalized_euc:.1f} g/dL</span>'
        
        # Display label and prediction value
        st.markdown(f'<span class="label">Haemoglobin ({label}) Original:</span><br>{display_value}</p>', unsafe_allow_html=True)
        st.markdown(f'<span class="label">Haemoglobin ({label}) Normalized Euclidean:</span><br>{display_value2}</p>', unsafe_allow_html=True)

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(wavelengths, absorbance_data.iloc[0], marker='o', linestyle='-', color='b')
    plt.xlabel('Wavelength (nm)', fontweight='bold', fontsize=14)
    plt.ylabel('Absorbance', fontweight='bold', fontsize=14)
    plt.xticks(rotation='vertical', fontweight='bold', fontsize=12)
    plt.yticks(fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)
    
if __name__ == "__main__":
    main()
