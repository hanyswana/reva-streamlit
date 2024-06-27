import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import requests, pytz
from scipy import sparse
from datetime import datetime
from sklearn.preprocessing import Normalizer, PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression


utc_now = datetime.now(pytz.utc)
singapore_time = utc_now.astimezone(pytz.timezone('Asia/Singapore'))
formatted_time = singapore_time.strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"Time: {formatted_time}")


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
    mean_corrected = input_data - np.mean(input_data, axis=1, keepdims=True)
    snv_transformed = mean_corrected / np.std(mean_corrected, axis=1, keepdims=True)
    return snv_transformed


def pds_transform(input_data, pds_model):
    F, a = pds_model
    transformed_data = input_data.dot(F) + a
    return transformed_data


def custom_transform(input_data, pds_models):
    transformed_data = np.zeros_like(input_data)

    for start, end, model in pds_models:
        slave_segment = input_data[:, start:end]
        transformed_segment = model.predict(slave_segment)
        transformed_data[:, start:end] = transformed_segment

    return transformed_data


def json_data():
    # API --------------------------------------------------------------------------------------------------------------------
    # First API call
    api_url1 = "https://x8ki-letl-twmt.n7.xano.io/api:5r4pCOor/bgdata_hb"
    payload1 = {}
    response1 = requests.get(api_url1, params=payload1)

    if response1.status_code == 200:
        data1 = response1.json()
    else:
        st.write("Error in first API call:", response1.status_code)
        return None

    # Second API call
    api_url2 = "https://x8ki-letl-twmt.n7.xano.io/api:UpqVw9TY/spectraldata_hb"
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
    absorbance_df = df2.div(df1.values).pow(0.5)
    # st.write(absorbance_df)


    # PREPROCESS ------------------------------------------------------------------------------------------------------------------
    # 1. SNV
    absorbance_snv = snv(absorbance_df.values)
    absorbance_snv_df = pd.DataFrame(absorbance_snv, columns=absorbance_df.columns)
    
    # # 2. Euclidean normalization
    # normalizer = Normalizer(norm='l2')  # Euclidean normalization
    # absorbance_normalized_euc = normalizer.transform(absorbance_snv_df)
    # absorbance_normalized_euc_df = pd.DataFrame(absorbance_normalized_euc, columns=absorbance_df.columns)

    # # 3. Manhattan normalization
    # normalizer = Normalizer(norm='l1')  # Manhattan normalization
    # absorbance_normalized_manh = normalizer.transform(absorbance_snv_df)
    # absorbance_normalized_manh_df = pd.DataFrame(absorbance_normalized_manh, columns=absorbance_df.columns)

    # 4. Baseline removal
    baseline_remover = BaselineRemover()
    absorbance_baseline_removed = baseline_remover.transform(absorbance_snv_df)
    absorbance_baseline_removed_df = pd.DataFrame(absorbance_baseline_removed, columns=absorbance_df.columns)

    # pds_model = joblib.load('pds_model_U11_snv_baseline.joblib')
    # with open('pds_model_U6_snv_baseline.pkl', 'rb') as f:
    #     pds_model = pickle.load(f)

    # absorbance_transformed = pds_transform(absorbance_baseline_removed_df.values, pds_model)
    # absorbance_transformed_df = pd.DataFrame(absorbance_transformed, columns=absorbance_df.columns)
    # absorbance_all_pp_df = absorbance_transformed_df

    absorbance_all_pp_df = absorbance_baseline_removed_df
    # st.write('19 preprocessed data :')
    # st.write(absorbance_all_pp_df)

    reference_file_path = 'Lablink_134_SNV_Baseline.csv'
    reference_df = pd.read_csv(reference_file_path, usecols=range(3, 22))
    reference_df = reference_df.apply(pd.to_numeric, errors='coerce')
    
    golden_values = reference_df.mean().values
    Min = reference_df.min().values
    Max = reference_df.max().values
 
    return absorbance_df, absorbance_all_pp_df, wavelengths, golden_values, Min, Max
    

def create_csv(golden_values, Min, Max, wavelengths):
    data = {
        'Wavelength': wavelengths,
        'Golden Values': golden_values,
        'Min': Min,
        'Max': Max
    }
    df = pd.DataFrame(data).T
    df.to_csv('golden_values_min_max.csv', index=False)
    # st.write(df)
    

def select_for_prediction(absorbance_df, selected_wavelengths):
    return absorbance_df[selected_wavelengths]


def load_model(model_dir):
    if model_dir.endswith('.tflite'):
        interpreter = tf.lite.Interpreter(model_path=model_dir)
        interpreter.allocate_tensors()
        return interpreter
    else:
        model = tf.saved_model.load(model_dir)
        return model


def predict_with_model(model, input_data):
    if isinstance(model, tf.lite.Interpreter):
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        # Ensure input data is 2D: [batch_size, num_features]
        input_data = input_data.values.astype('float32')
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)  # Reshape if single row input
        
        model.set_tensor(input_details[0]['index'], input_data)
        model.invoke()
        predictions = model.get_tensor(output_details[0]['index'])
        return predictions
    else:
        input_data = input_data.values.astype('float32').reshape(-1, 10)
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
        predictions = model(input_tensor)
        return predictions.numpy()


def main():

    model_paths_with_labels = [
        ('SNV + BR (R45)', 'Lablink_134_SNV_Baseline_pls_top_10.parquet_best_model_2024-05-09_20-22-34_R45_77%')
    ]
    
    absorbance_df, absorbance_all_pp_df, wavelengths, golden_values, Min, Max = json_data()

    create_csv(golden_values, Min, Max, wavelengths)
    
    for label, model_path in model_paths_with_labels:

        selected_wavelengths = ['_415nm', '_445nm', '_515nm', '_555nm', '_560nm', '_610nm', '_680nm', '_730nm', '_900nm', '_940nm'] # for API (SNV + BR / SNV + euc + BR) - new
        # selected_wavelengths = ['_445nm', '_515nm', '_555nm', '_560nm', '_585nm', '_610nm', '_680nm', '_730nm', '_900nm', '_940nm'] # for API (SNV + manh + BR) - new
        prediction_data = select_for_prediction(absorbance_all_pp_df, selected_wavelengths)
        # st.write('10 selected preprocessed data :')
        # st.write(prediction_data)
        
        model = load_model(model_path)
        predictions = predict_with_model(model, prediction_data)
        predictions_value = predictions[0][0]

        correlation = np.corrcoef(absorbance_all_pp_df.iloc[0], golden_values)[0, 1]

        Min = np.array(Min, dtype=float)
        Max = np.array(Max, dtype=float)
        absorbance_values = (absorbance_all_pp_df.values)

        out_of_range = (absorbance_values < Min) | (absorbance_values > Max)
        count_out_of_range = np.sum(out_of_range)
        total_values = absorbance_values.size
        in_range_percentage = 100 - ((count_out_of_range / total_values) * 100)

        st.markdown("""
        <style>
        .label {font-size: 20px; font-weight: bold; color: black;}
        .value {font-size: 40px; font-weight: bold; color: blue;}
        # .high-value {color: red;}
        </style> """, unsafe_allow_html=True)

        if predictions_value > 100:
            display_text = 'Above 100 g/dL'
        elif predictions_value < 0:
            display_text = 'Below 0 g/dL'
        else:
            display_text = f'{predictions_value:.1f} g/dL'
            
        display_value = f'<span class="value">{display_text}</span>'

        st.markdown(f'<span class="label">Haemoglobin ({label}):</span><br>{display_value}</p>', unsafe_allow_html=True)
        st.markdown(f'<span class="label">Similarity score:</span><br><span class="value">{in_range_percentage:.0f} %</span>', unsafe_allow_html=True)
        # st.markdown(f'<span class="label">Correlation:</span><br><span class="value">{correlation:.2f}</span>', unsafe_allow_html=True)

    # plt.figure(figsize=(10, 4))
    # plt.plot(wavelengths, absorbance_all_pp_df.iloc[0], marker='o', linestyle='-', color='b', label='Sample')
    # # plt.plot(wavelengths, absorbance_df.iloc[0], marker='o', linestyle='--', color='b', label='Raw sample')
    # plt.plot(wavelengths, Min, linestyle='--', color='r', label='Min')
    # plt.plot(wavelengths, Max, linestyle='--', color='y', label='Max')
    # plt.title('Absorbance', fontweight='bold', fontsize=20)
    # plt.xlabel('Wavelength (nm)', fontweight='bold', fontsize=14)
    # plt.ylabel('Absorbance', fontweight='bold', fontsize=14)
    # plt.xticks(rotation='vertical', fontweight='bold', fontsize=12)
    # plt.yticks(fontweight='bold', fontsize=12)
    # plt.tight_layout()
    # plt.legend()
    # plt.show()
    # st.pyplot(plt)


if __name__ == "__main__":
    main()
