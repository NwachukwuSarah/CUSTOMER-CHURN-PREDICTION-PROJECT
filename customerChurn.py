import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the model and preprocessing objects
@st.cache_resource
def load_model():
    with open("CCmodel.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle["CCmodel"], bundle["encoder"], bundle["columns"], bundle["scaler"]

modelUS, encoder, categorical_cols_for_encoder, scaler = load_model()

def predict(raw_user_inputs):
  input_df = pd.DataFrame([raw_user_inputs])

  columns_to_normalize_no = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                             'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
  for col in columns_to_normalize_no:
      if col == 'MultipleLines':
          input_df[col] = input_df[col].replace('No phone service', 'No')
      else:
          input_df[col] = input_df[col].replace('No internet service', 'No')


  # 2. Map Gender to 0 (Male) and 1 (Female)
  input_df['gender'] = input_df['gender'].map({'Male': 0, 'Female': 1})

  # 3. Map 'Yes' to 1 and 'No' to 0 for relevant binary columns
  binary_cols_to_map = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                        'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                        'DeviceProtection', 'TechSupport', 'StreamingTV',
                        'StreamingMovies', 'PaperlessBilling']
  for col in binary_cols_to_map:
      input_df[col] = input_df[col].map({'No': 0, 'Yes': 1})

  # Separate numerical and categorical features
  numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
  categorical_input_df = input_df[categorical_cols_for_encoder].copy()
  numerical_input_df = input_df[numerical_cols].copy()

  # One-hot encode categorical features
  encoded_categorical_features = encoder.transform(categorical_input_df)

  # Scale numerical features
  scaled_numerical_features = scaler.transform(numerical_input_df)

  # Concatenate encoded categorical features and scaled numerical features
  final_features = np.hstack((encoded_categorical_features, scaled_numerical_features))

  prediction_prob = modelUS.predict(final_features)[0][0]
  prediction_class = (prediction_prob > 0.5).astype(int)

  st.subheader('Prediction Result:')
  if prediction_class == 1:
      st.error(f'This customer is likely to CHURN! (Probability: {prediction_prob:.2f})')
  else:
      st.success(f'This customer is likely to stay. (Probability: {prediction_prob:.2f})')


# UI
st.title('Customer Churn Prediction')
st.write('Enter customer details to predict churn.')

st.sidebar.header('Customer Details')
raw_user_inputs = {}

gender_display_options = ['Female', 'Male']
yes_no_display_options = ['No', 'Yes']
multiple_lines_display_options = ['No phone service', 'No', 'Yes']
internet_service_display_options = ['DSL', 'Fiber optic', 'No']
online_service_display_options = ['No', 'Yes', 'No internet service']
contract_display_options = ['Month-to-month', 'One year', 'Two year']
payment_method_display_options = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']


raw_user_inputs['gender'] = st.sidebar.radio("What's your customer's Gender", gender_display_options, key='gender_ui')
raw_user_inputs['SeniorCitizen'] = st.sidebar.radio('Is the customer a Senior Citizen?', yes_no_display_options, key='seniorcitizen_ui')
raw_user_inputs['Partner'] = st.sidebar.radio('Does the customer have a Partner?', yes_no_display_options, key='partner_ui')
raw_user_inputs['Dependents'] = st.sidebar.radio('Does the customer have Dependents?', yes_no_display_options, key='dependents_ui')
raw_user_inputs['PhoneService'] = st.sidebar.radio('Does the customer have a Phone Service subscription?', yes_no_display_options, key='phoneservice_ui')
raw_user_inputs['MultipleLines'] = st.sidebar.selectbox('Deos the customer have Multiple Lines?', multiple_lines_display_options, key='multiplelines_ui')
raw_user_inputs['InternetService'] = st.sidebar.selectbox('Does the customer have an Internet Service?', internet_service_display_options, key='internetservice_ui')
raw_user_inputs['OnlineSecurity'] = st.sidebar.selectbox('Does the customer have Online Security subscription?', online_service_display_options, key='onlinesecurity_ui')
raw_user_inputs['OnlineBackup'] = st.sidebar.selectbox('Does the customer have Online Backup?', online_service_display_options, key='onlinebackup_ui')
raw_user_inputs['DeviceProtection'] = st.sidebar.selectbox('Does the customer have Device Protection subscription?>', online_service_display_options, key='deviceprotection_ui')
raw_user_inputs['TechSupport'] = st.sidebar.selectbox('Does the customer have Tech Support subscription?', online_service_display_options, key='techsupport_ui')
raw_user_inputs['StreamingTV'] = st.sidebar.selectbox('Does the customer have a Streaming TV subscription', online_service_display_options, key='streamingtv_ui')
raw_user_inputs['StreamingMovies'] = st.sidebar.selectbox('Does the customer Stream Movies?', online_service_display_options, key='streamingmovies_ui')
raw_user_inputs['Contract'] = st.sidebar.selectbox('What kind of contract does the customer have?', contract_display_options, key='contract_ui')
raw_user_inputs['PaperlessBilling'] = st.sidebar.radio('Does the customer use Paperless Billing method?', yes_no_display_options, key='paperlessbilling_ui')
raw_user_inputs['PaymentMethod'] = st.sidebar.selectbox('What kind of Payment Method does the customer use?', payment_method_display_options, key='paymentmethod_ui')
raw_user_inputs['tenure'] = st.sidebar.slider('Tenure (months)?', min_value=1.0, max_value=72.0, value=32.0, step=1.0, key='tenure_ui')
raw_user_inputs['MonthlyCharges'] = st.sidebar.number_input("What are the customer's Monthly Charges?", min_value=18.25, max_value=118.75, value=65.0, step=0.01, key='monthlycharges_ui')
raw_user_inputs['TotalCharges'] = st.sidebar.number_input("What are the customer's Total Charges?", min_value=18.80, max_value=8684.80, value=2000.0, step=0.01, key='totalcharges_ui')


if st.sidebar.button('Predict Churn'):
    try:
        predict(raw_user_inputs)
    except Exception as e:

        st.error(f"Prediction error: {str(e)}")


