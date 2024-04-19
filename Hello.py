import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import Normalizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES
from scipy import sparse
import numpy as np
from datetime import datetime
import pytz

# st.markdown("""
# <style>
# .custom-font {font-size: 16px; font-weight: bold;}
# </style> """, unsafe_allow_html=True)

# st.markdown('<p class="custom-font">Absorbance data :</p>', unsafe_allow_html=True)

utc_now = datetime.now(pytz.utc)
singapore_time = utc_now.astimezone(pytz.timezone('Asia/Singapore'))
formatted_time = singapore_time.strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"Time: {formatted_time}")

# Custom Baseline Removal Transformer
class BaselineRemover(TransformerMixin, BaseEstimator):
    def __init__(self, *, copy=True):
        self.copy = copy

    def fit(self, X, y=None):
        if sparse.issparse(X):
            raise ValueError('Sparse matrices not supported!')
        return self

    def transform(self, X, copy=None):
        copy = copy if copy is not None else self.copy
        X = self._validate_data(X, reset=True, accept_sparse='csr', copy=copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite='allow-nan')
        X = self.remove_baseline(X.T).T
        return X

    def remove_baseline(self, spectra):
        return spectra - spectra.mean(axis=0)

    def _more_tags(self):
        return {'allow_nan': True}

def snv(input_data):
    # Mean centering and scaling by standard deviation for each spectrum
    mean_corrected = input_data - np.mean(input_data, axis=1, keepdims=True)
    snv_transformed = mean_corrected / np.std(mean_corrected, axis=1, keepdims=True)
    return snv_transformed
        
def json_data():
    # First API call
    api_url1 = "https://x8ki-letl-twmt.n7.xano.io/api:U4wk_Gn6/BackgroundReading"
    payload1 = {}
    response1 = requests.get(api_url1, params=payload1)

    if response1.status_code == 200:
        data1 = response1.json()
    else:
        st.write("Error in first API call:", response1.status_code)
        return None

    # Second API call
    api_url2 = "https://x8ki-letl-twmt.n7.xano.io/api:DKaWNKM4/spectral_data"
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
    # st.write(df1)
    # st.write(df2)
    wavelengths = df1.columns

    # Element-wise division of the dataframes & convert absorbance data to csv
    absorbance_df = df1.div(df2.values).pow(2)

    # Selected wavelengths based on user requirement
    # wavelengths = ['415nm', '445nm', '480nm', '515nm', '555nm', '585nm', '590nm', '610nm', '630nm', '730nm']
    # absorbance_df = absorbance_df[wavelengths]
    # st.write(absorbance_df)

    # Apply SNV to the absorbance data after baseline removal
    absorbance_snv = snv(absorbance_df.values)
    absorbance_snv_df = pd.DataFrame(absorbance_snv, columns=absorbance_df.columns)
    # st.write('SNV Transformation')
    # st.write(absorbance_snv_df)
    
    # # Normalize the absorbance data using Euclidean normalization
    # normalizer = Normalizer(norm='l2')  # Euclidean normalization
    # absorbance_normalized_euc = normalizer.transform(absorbance_df)
    # absorbance_normalized_euc_df = pd.DataFrame(absorbance_normalized_euc, columns=absorbance_df.columns)
    # st.write('Euclidean absorbance')
    # st.write(absorbance_normalized_euc_df)

    # # Convert normalized DataFrame to CSV (optional step, depending on your needs)
    # absorbance_normalized_euc_df.to_csv('absorbance_data_normalized_euc.csv', index=False)

    # # Normalize the absorbance data using Manhattan normalization
    # normalizer = Normalizer(norm='l1')  # Manhattan normalization
    # absorbance_normalized_manh = normalizer.transform(absorbance_df)
    # absorbance_normalized_manh_df = pd.DataFrame(absorbance_normalized_manh, columns=absorbance_df.columns)
    # st.write('Manhattan absorbance')
    # st.write(absorbance_normalized_manh_df)

    # # Convert normalized DataFrame to CSV (optional step, depending on your needs)
    # absorbance_normalized_manh_df.to_csv('absorbance_data_normalized_manh.csv', index=False)

    # Apply baseline removal to the absorbance data
    baseline_remover = BaselineRemover()
    absorbance_baseline_removed = baseline_remover.transform(absorbance_df)
    absorbance_baseline_removed_df = pd.DataFrame(absorbance_baseline_removed, columns=absorbance_df.columns)
    # st.write('Baseline removal')
    # st.write(absorbance_baseline_removed_df)
    
    absorbance_snv_baseline_removed = baseline_remover.transform(absorbance_snv)
    absorbance_snv_baseline_removed_df = pd.DataFrame(absorbance_snv_baseline_removed, columns=absorbance_df.columns)
    # st.write('SNV + BR')
    # st.write(absorbance_snv_baseline_removed_df)

    # First row of absorbance data
    absorbance_data = absorbance_df.iloc[0]  
 
    return absorbance_df, absorbance_snv_baseline_removed_df, wavelengths
    # return absorbance_df, wavelengths

def select_for_prediction(absorbance_data, selected_wavelengths):
    return absorbance_data[selected_wavelengths]
    
def load_model(model_dir):
    if model_dir.endswith('.tflite'):  # Check if model is a TensorFlow Lite model
        # Load TensorFlow Lite model
        interpreter = tf.lite.Interpreter(model_path=model_dir)
        interpreter.allocate_tensors()
        return interpreter
    else:
        # Load TensorFlow SavedModel
        model = tf.saved_model.load(model_dir)
        return model

# def predict_with_model(model, input_data):
#     if isinstance(model, tf.lite.Interpreter):  # Check if model is TensorFlow Lite Interpreter
#         input_details = model.get_input_details()
#         output_details = model.get_output_details()
        
#         input_data = input_data.astype('float32')
#         input_data = np.expand_dims(input_data, axis=0)
        
#         # Assuming input_data is already in the correct shape and type
#         model.set_tensor(input_details[0]['index'], input_data)
#         model.invoke()
#         predictions = model.get_tensor(output_details[0]['index'])
#         return predictions  # This will be a numpy array
#     else:
#         # Existing prediction code for TensorFlow SavedModel
#         input_array = input_data.to_numpy(dtype='float32')
#         input_array_reshaped = input_array.reshape(-1, 10)  # Adjust to match the number of features your model expects
#         input_tensor = tf.convert_to_tensor(input_array_reshaped, dtype=tf.float32)
#         predictions = model(input_tensor)
#         return predictions.numpy()  # Convert predictions to numpy array if needed

def predict_with_model(model, input_data):
    if isinstance(model, tf.lite.Interpreter):  # Check if model is TensorFlow Lite Interpreter
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        # Ensure input data is 2D: [batch_size, num_features]
        input_data = input_data.values.astype('float32')  # Convert DataFrame to numpy and ensure dtype
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)  # Reshape if single row input
        
        model.set_tensor(input_details[0]['index'], input_data)
        model.invoke()
        predictions = model.get_tensor(output_details[0]['index'])
        return predictions  # This will be a numpy array
    else:
        # Assuming TensorFlow SavedModel prediction logic
        input_data = input_data.values.astype('float32').reshape(-1, 10)  # Adjust based on your model's expected input
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
        predictions = model(input_tensor)
        return predictions.numpy()

def main():
    # Define model paths with labels
    model_paths_with_labels = [
        ('TF', 'snv_baseline_removed_pls_top_10_float32.parquet_best_model_2024-04-03_04-18-56'),
        ('TFL', 'tflite_model_snv_br_10_2024-04-03_04-18-56.tflite'),
        ('TFL-q', 'tflite_model_snv_br_10_quant_2024-04-03_04-18-56.tflite')
    ]

    # Get data from server (simulated here)
    absorbance_df, absorbance_snv_baseline_removed_df, wavelengths = json_data()
    # absorbance_data, wavelengths = json_data()

    for label, model_path in model_paths_with_labels:

        selected_wavelengths = ['415nm', '445nm', '480nm', '515nm', '555nm', '585nm', '590nm', '610nm', '630nm', '730nm']
        prediction_data = select_for_prediction(preprocess_data, selected_wavelengths)
        
        model = load_model(model_path)
        # st.write(model)
        
        # # Predict with original absorbance data
        # predictions_original = predict_with_model(model, absorbance_df)
        # predictions_value_original = predictions_original[0][0]
        
        # # Predict with Euclidean normalized absorbance data
        # predictions_normalized_euc = predict_with_model(model, absorbance_normalized_euc_data)
        # predictions_value_normalized_euc = predictions_normalized_euc[0][0]

        # # Predict with Manhattan normalized absorbance data
        # predictions_normalized_manh = predict_with_model(model, absorbance_normalized_manh_data)
        # predictions_value_normalized_manh = predictions_normalized_manh[0][0]

        # # Predict with baseline removed absorbance data
        # predictions_baseline_removed = predict_with_model(model, absorbance_baseline_removed_data)
        # predictions_value_baseline_removed = predictions_baseline_removed[0][0]

        # # Predict with SNV transformed absorbance data
        # predictions_snv = predict_with_model(model, absorbance_snv_data)
        # predictions_value_snv = predictions_snv[0][0]

        # Predict with SNV and BR transformed absorbance data
        # predictions_snv_baseline_removed = predict_with_model(model, absorbance_snv_baseline_removed_df)
        predictions_snv_baseline_removed = predict_with_model(model, prediction_data)
        predictions_value_snv_baseline_removed = predictions_snv_baseline_removed[0][0]

    
        st.markdown("""
        <style>
        .label {font-size: 20px; font-weight: bold; color: black;}
        .value {font-size: 40px; font-weight: bold; color: blue;}
        # .high-value {color: red;}
        </style> """, unsafe_allow_html=True)

        if predictions_value_snv_baseline_removed > 15:
            display_text = 'Above 15 g/dL'
        elif predictions_value_snv_baseline_removed < 10.9:
            display_text = 'Below 11 g/dL'
        else:
            display_text = f'{predictions_value_snv_baseline_removed:.1f} g/dL'
            
        # Format the display value with consistent styling
        display_value6 = f'<span class="value">{display_text}</span>'
                
        # # Add condition for prediction value
        # if predictions_value_snv_baseline_removed > 25:
        #     # display_value = f'<span class="high-value">High value : ({predictions_value_original:.1f} g/dL)</span>'
        #     # display_value2 = f'<span class="high-value">High value : ({predictions_value_normalized_euc:.1f} g/dL)</span>'
        #     # display_value3 = f'<span class="high-value">High value : ({predictions_value_normalized_manh:.1f} g/dL)</span>'
        #     # display_value4 = f'<span class="high-value">High value : ({predictions_value_baseline_removed:.1f} g/dL)</span>'
        #     # display_value5 = f'<span class="high-value">High value : ({predictions_value_snv:.1f} g/dL)</span>'
        #     display_value6 = f'<span class="high-value">High value : ({predictions_value_snv_baseline_removed:.1f} g/dL)</span>'
        # else:
        #     # display_value = f'<span class="value">{predictions_value_original:.1f} g/dL</span>'
        #     # display_value2 = f'<span class="value">{predictions_value_normalized_euc:.1f} g/dL</span>'
        #     # display_value3 = f'<span class="value">{predictions_value_normalized_manh:.1f} g/dL</span>'
        #     # display_value4 = f'<span class="value">{predictions_value_baseline_removed:.1f} g/dL</span>'
        #     # display_value5 = f'<span class="value">{predictions_value_snv:.1f} g/dL</span>'
        #     display_value6 = f'<span class="value">{predictions_value_snv_baseline_removed:.1f} g/dL</span>'
        
        # # Display label and prediction value
        # st.markdown(f'<span class="label">Haemoglobin :</span><br>{display_value}</p>', unsafe_allow_html=True)
        # st.markdown(f'<span class="label">Haemoglobin ({label}) Normalized Euclidean:</span><br>{display_value2}</p>', unsafe_allow_html=True)
        # st.markdown(f'<span class="label">Haemoglobin ({label}) Normalized Manhattan:</span><br>{display_value3}</p>', unsafe_allow_html=True)
        # st.markdown(f'<span class="label">Haemoglobin ({label}) Baseline removal:</span><br>{display_value4}</p>', unsafe_allow_html=True)
        # st.markdown(f'<span class="label">Haemoglobin ({label}) SNV:</span><br>{display_value5}</p>', unsafe_allow_html=True)
        st.markdown(f'<span class="label">Haemoglobin ({label}):</span><br>{display_value6}</p>', unsafe_allow_html=True)


    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(wavelengths, absorbance_df.iloc[0], marker='o', linestyle='-', color='b')
    plt.xlabel('Wavelength (nm)', fontweight='bold', fontsize=14)
    plt.ylabel('Absorbance', fontweight='bold', fontsize=14)
    plt.xticks(rotation='vertical', fontweight='bold', fontsize=12)
    plt.yticks(fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)
    
if __name__ == "__main__":
    main()
