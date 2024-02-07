import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import json

# Load the trained model
model_path = 'C:\\Users\\parkway\\Documents\\Python Code\\LP_Models\\LP_Models\\models\\dp_model.h5'
model = tf.keras.models.load_model(model_path)

def preprocess_data(data):
    # Your preprocessing code here
    columns_to_drop = ['org_id', 'user_id', 'status_id', 'loan_id', 'work_start_date', 'work_email', 'loan_request_day',
                       'current_employer', 'work_email_validated', 'first_account', 'last_account', 'created_on',
                       'process_time', 'photo_url', 'logins']

    new_data = data.drop(columns=columns_to_drop)
    new_data["requested_amount"] = new_data["requested_amount"].astype(int)

    columns_to_encode = ['gender', 'marital_status', 'type_of_residence', 'educational_attainment',
                         'sector_of_employment', 'monthly_net_income', 'country', 'city', 'lga', 'purpose',
                         'selfie_bvn_check', 'selfie_id_check', 'device_name', 'mobile_os', 'os_version',
                         'no_of_dependent', 'employment_status']

    label_encoder = LabelEncoder()
    for column in columns_to_encode:
        new_data[column] = label_encoder.fit_transform(new_data[column])

    def process_column(column):
        new_column = []
        for value in column:
            if value.endswith("days"):
                new_value = int(value[:-5])
            elif value.endswith("months"):
                new_value = int(value[:-6]) * 30
            elif value.endswith("weeks"):
                new_value = int(value[:-5]) * 7
            else:
                new_value = 1
            new_column.append(new_value)
        return new_column

    new_data['proposed_payday'] = process_column(new_data['proposed_payday'])

    return new_data

# Streamlit app
def main():
    st.title('Loan Default Prediction')

    # Input JSON data
    json_data = st.text_area('Enter JSON data:', '{"your": "json", "data": "here"}')

    # Button to prettify JSON
    prettify_button = st.button('Prettify JSON')

    if prettify_button:
        try:
            # Prettify JSON
            prettified_json = json.dumps(json.loads(json_data), indent=2)
            st.text_area('Prettified JSON:', prettified_json, height=200)
        except json.JSONDecodeError:
            st.error('Invalid JSON format. Please enter a valid JSON.')

    # Button to make prediction
    predict_button = st.button('Predict')

    if predict_button:
        try:
            # Convert JSON to dictionary
            data = json.loads(json_data)

            # Preprocess the input data
            input_data = pd.DataFrame(data)
            preprocessed_data = preprocess_data(input_data)

            # Display preprocessed data for inspection
            st.subheader('Preprocessed Data:')
            st.dataframe(preprocessed_data)

            # Make predictions using the model
            # predictions = model.predict(preprocessed_data.values)
            input_array = preprocessed_data.values.reshape(1, -1)  # Reshape the input array
            predictions = model.predict(input_array)

            # Adjust the prediction output format and round the prediction to 0 or 1
            rounded_prediction = int(np.round(predictions[0]))

            # Determine the result based on the rounded prediction
            if rounded_prediction == 0:
                result = {'prediction': float(predictions), 'rounded prediction': 0, 'status': 'Default'}
            else:
                result = {'prediction': float(predictions), 'rounded prediction': 1, 'status': 'Not Default'}

            # Display prediction result
            st.subheader('Prediction Result:')
            st.json(result)

        except json.JSONDecodeError:
            st.error('Invalid JSON format. Please enter a valid JSON.')


if __name__=='__main__': 
    main()
