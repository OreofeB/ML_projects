from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS  # Add this import for CORS support

app = Flask(__name__)
api = Api(app)
CORS(app)  # Enable CORS

# Load the trained model
# Assume you have a trained model stored in a variable named 'model'
# model_path = 'C:\\Users\\parkway\\Documents\\Python Code\\LP_Models\\models\\dp_model.pkl'
# model = pickle.load(open(model_path, 'rb'))  # Load your model here
model_path = 'C:\\Users\\parkway\\Documents\\Python Code\\LP_Models\\models\\dp_model.h5'
model = tf.keras.models.load_model(model_path)

def preprocess_data(data):
    # Your preprocessing code here
    columns_to_drop = ['org_id', 'user_id', 'status_id', 'loan_id', 'work_start_date', 'work_email', 'loan_request_day',
                       'current_employer', 'work_email_validated', 'first_account', 'last_account', 'created_on',
                       'process_time', 'photo_url', 'logins']

    print("Columns in DataFrame:", data.columns)
    new_data = data.drop(columns=columns_to_drop)
    # new_data = new_data.dropna()
    # new_data = new_data[new_data['status_id'] != 1]

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

# Define a resource for your loan prediction
class LoanPrediction(Resource):
    def post(self):
        # Get data from the request
        data = request.get_json()

        # Preprocess the input data
        input_data = pd.DataFrame(data)
        preprocessed_data = preprocess_data(input_data)

        # Print preprocessed data for inspection
        print("Preprocessed Data Columns:", preprocessed_data.columns)
        print("Preprocessed Data:", preprocessed_data)

        # Make predictions using the model
        predictions = model.predict(preprocessed_data.values)

        # Adjust the prediction output format and Round the prediction to 0 or 1
        rounded_prediction = int(np.round(predictions[0]))

        # Determine the result based on the rounded prediction
        if rounded_prediction == 0:
            result = {'prediction': float(predictions), 'rounded prediction': 0, 'status': 'Default'}
        else:
            result = {'prediction': float(predictions), 'rounded prediction': 1, 'status': 'Not Default'}


        return jsonify(result)

# Add the resource to the API with the specified endpoint
api.add_resource(LoanPrediction, '/predict')

if __name__ == '__main__':
    app.run(debug=True)
