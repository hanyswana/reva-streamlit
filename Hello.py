import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import Normalizer
from sklearn.utils.validation import FLOAT_DTYPES
from scipy import sparse

# # Set page configuration to wide mode
# st.set_page_config(layout="wide")

# st.markdown("""
# <style>
# .custom-font {font-size: 16px; font-weight: bold;}
# </style> """, unsafe_allow_html=True)

# st.markdown('<p class="custom-font">Absorbance data :</p>', unsafe_allow_html=True)


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
    api_url1 = "https://x8ki-letl-twmt.n7.xano.io/api:3Ws6ADLi/bgdata"
    payload1 = {}
    response1 = requests.get(api_url1, params=payload1)

    if response1.status_code == 200:
        data1 = response1.json()
    else:
        st.write("Error in first API call:", response1.status_code)
        return None

    # Second API call
    api_url2 = "https://x8ki-letl-twmt.n7.xano.io/api:Qc5crfn2/spectraldata"
    payload2 = {}
    response2 = requests.get(api_url2, params=payload2)

    if response2.status_code == 200:
        data2 = response2.json()
    else:
        st.write("Error in second API call:", response2.status_code)
        return None

    # Extract first line of data from both API responses and convert to numeric
    df1 = pd.DataFrame(data1).iloc[:1].apply(pd.to_numeric, errors='coerce')
    df2 = pd.DataFrame(data2).iloc[:5].apply(pd.to_numeric, errors='coerce')
    wavelengths = df1.columns
    st.write('Background')
    st.write(df1)
    st.write('Spectral')
    st.write(df2)

    # # Element-wise division of the dataframes & convert absorbance data to csv
    # # absorbance_df = df1.div(df2.values).pow(2)
    # absorbance_df = df1.div(row.values, axis='columns').pow(2)
    # st.write('Original absorbance')
    # st.write(absorbance_df)

    # # Apply SNV to the absorbance data
    # absorbance_snv = snv(absorbance_df.values)
    # absorbance_snv_df = pd.DataFrame(absorbance_snv, columns=absorbance_df.columns)
    # st.write('SNV Transformation')
    # st.write(absorbance_snv_df)

    # # Apply baseline removal to the absorbance data
    # baseline_remover = BaselineRemover()
    # absorbance_baseline_removed = baseline_remover.transform(absorbance_df)
    # absorbance_baseline_removed_df = pd.DataFrame(absorbance_baseline_removed, columns=absorbance_df.columns)
    # absorbance_snv_baseline_removed = baseline_remover.transform(absorbance_snv)
    # absorbance_snv_baseline_removed_df = pd.DataFrame(absorbance_snv_baseline_removed, columns=absorbance_df.columns)
    # st.write('Baseline removal')
    # st.write(absorbance_baseline_removed_df)
    # st.write('SNV + baseline removal')
    # st.write(absorbance_snv_baseline_removed_df)

    # # Normalize the absorbance data using Euclidean normalization
    # normalizer_euc= Normalizer(norm='l2')  # Euclidean normalization
    # absorbance_normalized_euc = normalizer_euc.transform(absorbance_df)
    # absorbance_normalized_euc_df = pd.DataFrame(absorbance_normalized_euc, columns=absorbance_df.columns)
    # absorbance_snv_normalized_euc = normalizer_euc.transform(absorbance_snv)
    # absorbance_snv_normalized_euc_df = pd.DataFrame(absorbance_snv_normalized_euc, columns=absorbance_df.columns)
    # st.write('Euc')
    # st.write(absorbance_normalized_euc_df)
    # st.write('SNV + euc')
    # st.write(absorbance_snv_normalized_euc_df)

    # # # Convert normalized DataFrame to CSV (optional step, depending on your needs)
    # # absorbance_normalized_euc_df.to_csv('absorbance_data_normalized_euc.csv', index=False)

    # # Normalize the absorbance data using Manhattan normalization
    # normalizer_manh = Normalizer(norm='l1')  # Manhattan normalization
    # absorbance_normalized_manh = normalizer_manh.transform(absorbance_df)
    # absorbance_normalized_manh_df = pd.DataFrame(absorbance_normalized_manh, columns=absorbance_df.columns)
    # absorbance_snv_normalized_manh = normalizer_manh.transform(absorbance_snv)
    # absorbance_snv_normalized_manh_df = pd.DataFrame(absorbance_snv_normalized_manh, columns=absorbance_df.columns)
    # st.write('Manh absorbance')
    # st.write(absorbance_normalized_manh_df)
    # st.write('SNV + manh')
    # st.write(absorbance_snv_normalized_manh_df)


    # # # First row of absorbance data
    # # absorbance_data = absorbance_df.iloc[0]  

    all_processed_dfs = []  # This will hold tuples of all processed versions of each df
    
    for index, row in df2.iterrows():
        absorbance_df = df1.div(row.values, axis='columns').pow(2)

        st.write(f'Original absorbance for row {index}')
        st.write(absorbance_df)
        
        # Apply SNV
        absorbance_snv = snv(absorbance_df.values)
        absorbance_snv_df = pd.DataFrame(absorbance_snv, columns=absorbance_df.columns)
        st.write('SNV')
        st.write(absorbance_snv_df)
            
        # Apply baseline removal
        baseline_remover = BaselineRemover()
        absorbance_baseline_removed = baseline_remover.transform(absorbance_df.values)
        absorbance_baseline_removed_df = pd.DataFrame(absorbance_baseline_removed, columns=absorbance_df.columns)
        absorbance_snv_baseline_removed = baseline_remover.transform(absorbance_snv)
        absorbance_snv_baseline_removed_df = pd.DataFrame(absorbance_snv_baseline_removed, columns=absorbance_df.columns)
        st.write('Baseline removal')
        st.write(absorbance_baseline_removed_df)
        st.write('SNV + baseline removal')
        st.write(absorbance_snv_baseline_removed_df)
        
        # Normalize the absorbance data using Euclidean normalization
        normalizer_euc = Normalizer(norm='l2')
        absorbance_normalized_euc = normalizer_euc.transform(absorbance_df)
        absorbance_normalized_euc_df = pd.DataFrame(absorbance_normalized_euc, columns=absorbance_df.columns)
        absorbance_snv_normalized_euc = normalizer_euc.transform(absorbance_snv)
        absorbance_snv_normalized_euc_df = pd.DataFrame(absorbance_snv_normalized_euc, columns=absorbance_df.columns)
        st.write('Euc')
        st.write(absorbance_normalized_euc_df)
        st.write('SNV + euc')
        st.write(absorbance_snv_normalized_euc_df)
        
        # Normalize the absorbance data using Manhattan normalization
        normalizer_manh = Normalizer(norm='l1')
        absorbance_normalized_manh = normalizer_manh.transform(absorbance_df)
        absorbance_normalized_manh_df = pd.DataFrame(absorbance_normalized_manh, columns=absorbance_df.columns)
        absorbance_snv_normalized_manh = normalizer_manh.transform(absorbance_snv)
        absorbance_snv_normalized_manh_df = pd.DataFrame(absorbance_snv_normalized_manh, columns=absorbance_df.columns)
        st.write('Manh absorbance')
        st.write(absorbance_normalized_manh_df)
        st.write('SNV + manh')
        st.write(absorbance_snv_normalized_manh_df)
        
        # Collect all processed versions for this division result
        processed_versions = (absorbance_snv_df, absorbance_baseline_removed_df, absorbance_snv_baseline_removed_df,
                              absorbance_normalized_euc_df, absorbance_snv_normalized_euc_df,
                              absorbance_normalized_manh_df, absorbance_snv_normalized_manh_df)
        
        all_processed_dfs.append(processed_versions)

    return all_processed_dfs, wavelengths
        
    # return absorbance_df, absorbance_normalized_euc_df, absorbance_snv_normalized_euc_df, absorbance_normalized_manh_df, absorbance_snv_normalized_manh_df, absorbance_baseline_removed_df, absorbance_snv_baseline_removed_df, absorbance_snv_df, wavelengths


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


def predict_with_model(model, input_data):
    if isinstance(model, tf.lite.Interpreter):  # Check if model is TensorFlow Lite Interpreter
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        input_data = input_data.astype('float32')
        input_data = np.expand_dims(input_data, axis=0)
        
        # Assuming input_data is already in the correct shape and type
        model.set_tensor(input_details[0]['index'], input_data)
        model.invoke()
        predictions = model.get_tensor(output_details[0]['index'])
        return predictions  # This will be a numpy array
    else:
        # Existing prediction code for TensorFlow SavedModel
        input_array = input_data.to_numpy(dtype='float32')
        input_array_reshaped = input_array.reshape(-1, 10)  # Adjust to match the number of features your model expects
        input_tensor = tf.convert_to_tensor(input_array_reshaped, dtype=tf.float32)
        predictions = model(input_tensor)
        return predictions.numpy()  # Convert predictions to numpy array if needed

def main():

    # model_paths_with_labels = [
    #     ('Ori (R39)', 'reva-lablink-hb-125-(original-data).csv_r2_0.39_2024-02-15_11-55-27')
    # ]

    model_paths_with_labels = [
        ('TF (SNV + BR 24-04-01)', 'snv_baseline_removed_pls_top_10_float32.parquet_best_model_2024-03-31_13-29-57'),
        ('TFL', 'tflite_model_snv_br_10.tflite'),
        ('TFL Q', 'tflite_model_snv_br_10_quant.tflite'),
        ('TF (SNV + BR 24-04-03)', 'snv_baseline_removed_pls_top_10_float32.parquet_best_model_2024-04-03_04-18-56'),
        ('TFL', 'tflite_model_snv_br_10_2024-04-03_04-18-56.tflite'),
        ('TFL Q', 'tflite_model_snv_br_10_quant_2024-04-03_04-18-56.tflite'),
        ('TF (SNV + euc 24-04-03)', 'snv_normalized_euclidean_pls_top_10_float32.parquet_best_model_2024-03-30_02-03-57'),
        ('TFL', 'tflite_model_snv_euc_10_2024-03-30_02-03-57.tflite'),
        ('TFL Q', 'tflite_model_snv_euc_10_quant_2024-03-30_02-03-57.tflite'),
        ('TF (SNV + manh 24-04-03)', 'snv_normalized_manhattan_pls_top_10_float32.parquet_best_model_2024-04-01_08-57-51'),
        ('TFL', 'tflite_model_snv_manh_10_2024-04-01_08-57-51.tflite'),
        ('TFL Q', 'tflite_model_snv_manh_10_quant_2024-04-01_08-57-51.tflite')
    ]

    # Assuming df1 and df2 are your dataframes obtained from API or other sources
    all_processed_dfs, wavelengths = json_data()

    results = []  # To accumulate prediction results

    # Example prediction loop, adapt to your prediction logic
    for model_label, model_path in model_paths_with_labels:
        model = load_model(model_path)
        
        for df_index, processed_versions in enumerate(all_processed_dfs):
            for preprocess_label, df in zip(["SNV", "BR", "SNV + BR", "Euc", "SNV + Euc", "Manh", "SNV + Manh"], processed_versions):
                for index, row in df.iterrows():
                    predictions = predict_with_model(model, row)
                    # st.write(f"Model: {model_label}, Preprocess: {preprocess_label}, Data Point: DF{df_index+1}-Row{index+1}, Prediction: {predictions}")

                    prediction_value = predictions[0][0]  # Assuming single value predictions for simplicity
                    formatted_prediction_value = f"{prediction_value:.1f}"
                    
                    # Append each prediction result to the results list
                    results.append({
                        "Model": model_label,
                        "Preprocessing": preprocess_label,
                        "Data Point": f"Sample {df_index+1}",
                        "Prediction (g/dL)": formatted_prediction_value
                    })

     # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results)

    st.markdown("""
    <style>
    /* This CSS selector targets the table elements in Streamlit */
    .stTable, .stDataFrame {
        font-size: 20px;  /* Increase font size */
        padding: 20px;    /* Add more padding */
    }
    </style>
    """, unsafe_allow_html=True)

    # Display the results as a table
    st.dataframe(results_df, height=500, width=700)


    # # Assuming json_data returns a tuple of all dataframes + wavelengths at the end
    # data = json_data()
    # if data is None:
    #     st.write("Failed to fetch or process data.")
    #     return

    # # Unpack all the DataFrames and wavelength from data
    # (absorbance_df, absorbance_normalized_euc_df, absorbance_snv_normalized_euc_df, 
    # absorbance_normalized_manh_df, absorbance_snv_normalized_manh_df, 
    # absorbance_baseline_removed_df, absorbance_snv_baseline_removed_df, 
    # absorbance_snv_df, wavelengths) = data

    # # Dictionary to hold DataFrame references and their labels for easy access
    # data_frames_with_labels = [
    #     ("Ori", absorbance_df),
    #     ("Euc", absorbance_normalized_euc_df),
    #     ("SNV + Euc", absorbance_snv_normalized_euc_df),
    #     ("Manh", absorbance_normalized_manh_df),
    #     ("SNV + Manh", absorbance_snv_normalized_manh_df),
    #     ("BR", absorbance_baseline_removed_df),
    #     ("SNV + BR", absorbance_snv_baseline_removed_df),
    #     ("SNV", absorbance_snv_df)
    # ]

    # results = []  # To accumulate prediction results
    

    # # Accumulate results in a list
    # for label, model_path in model_paths_with_labels:
    #     model = load_model(model_path)
        
    #     for preprocess_label, df in data_frames_with_labels:
    #         row = df.iloc[0]  # Assuming predictions on the first row
    #         predictions = predict_with_model(model, row)
    #         predictions_value = predictions[0][0]  # Assuming single value predictions

    #         formatted_predictions_value = f"{predictions_value:.1f}"


    #         # Append each prediction result to the results list
    #         results.append({
    #             "Model": label,
    #             "Preprocessing": preprocess_label,
    #             "Prediction (g/dL)": formatted_predictions_value
    #         })

    # # Convert the results list to a DataFrame
    # results_df = pd.DataFrame(results)

    # # Loop through each model and preprocessing combination
    # for label, model_path in model_paths_with_labels:
    #     model = load_model(model_path)
        
    #     for preprocess_label, df in data_frames_with_labels:
    #         for index, row in df.iterrows():  # Loop through each data row
    #             predictions = predict_with_model(model, row)
    #             predictions_value = predictions[0][0]  # Assuming single value predictions for simplicity
    
    #             formatted_predictions_value = f"{predictions_value:.1f}"
    
    #             # Append each prediction result to the results list
    #             # Including 'Data Point' to track which row the prediction corresponds to
    #             results.append({
    #                 "Model": label,
    #                 "Preprocessing": preprocess_label,
    #                 "Data Point": index + 1,  # Adjust index to be human-readable (starting at 1)
    #                 "Prediction (g/dL)": formatted_predictions_value
    #             })
    
    # # Convert the results list to a DataFrame
    # results_df = pd.DataFrame(results)

    # st.markdown("""
    # <style>
    # /* This CSS selector targets the table elements in Streamlit */
    # .stTable, .stDataFrame {
    #     font-size: 20px;  /* Increase font size */
    #     padding: 20px;    /* Add more padding */
    # }
    # </style>
    # """, unsafe_allow_html=True)

    # # Display the results as a table
    # st.dataframe(results_df, height=500, width=700)

    # # Loop through each model
    # for label, model_path in model_paths_with_labels:
    #     model = load_model(model_path)
    
    # for i, (label, model_path) in enumerate(model_paths_with_labels, start=1):
    #     # Dynamic section title with model label
    #     st.markdown(f"### Model {i}: {label}")
    #     st.markdown("---")  # Markdown for horizontal line
        
    #     model = load_model(model_path)
        
    #     # Loop through each preprocessing type
    #     for preprocess_label, df in data_frames_with_labels:
    #         # Making predictions for the first row as an example
    #         row = df.iloc[0]  # Assuming we are making predictions on the first row
    #         predictions = predict_with_model(model, row)
    #         predictions_value = predictions[0][0]  # Assuming each prediction returns a single value

    #         # Print the preprocessing label and prediction
    #         st.markdown(f"PP: {preprocess_label} | {label}<span style='color: blue;'> - <strong>Hb: {predictions_value:.1f} g/dL", unsafe_allow_html=True)
    

    # # Assuming json_data returns a tuple of all dataframes + wavelengths at the end
    # data_frames, wavelengths = json_data()[:-1], json_data()[-1]

    # for label, model_path in model_paths_with_labels:
    #     model = load_model(model_path)
        
    #     for df in data_frames:
    #         for index, row in df.iterrows():
    #             predictions = predict_with_model(model, row)
    #             predictions_value = predictions[0][0]  # Adjust based on your model's output
                
    #             # Display logic
    #             display_value = f"{predictions_value:.1f} g/dL"  # Example formatting
    #             st.write(f"Haemoglobin ({label}) - Sample {index+1}: {display_value}")

    
if __name__ == "__main__":
    main()
