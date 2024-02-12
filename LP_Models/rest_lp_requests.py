from fastapi import FastAPI, HTTPException, Depends
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime as dt
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# Enable CORS
origins = ["*"]  # Adjust this to your specific needs
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model_path = 'C:\\Users\\parkway\\Documents\\Python Code\\ML_Models\\LP_Models\\models\\dp_model.h5'
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

# Define a resource for your loan prediction
class LoanPrediction:
    def __init__(self, data: dict):
        self.data = pd.DataFrame(data)

    def predict_loan(self):
        # Preprocess the input data
        preprocessed_data = preprocess_data(self.data)

        # Print preprocessed data for inspection
        print("Preprocessed Data Columns:", preprocessed_data.columns)
        print("Preprocessed Data:", preprocessed_data)

        # Make predictions using the model
        predictions = model.predict(preprocessed_data.values)

        # Adjust the prediction output format and Round the prediction to 0 or 1
        rounded_prediction = int(np.round(predictions[0]))
        predictions_percentage = predictions * 100

        # Adjust the prediction output format and round the prediction to 0 or 1
        rounded_percentage = np.round(float(predictions_percentage), decimals=2)

        # Timestamp
        timestamp = dt.now().strftime("%Y-%m-%d %H:%M:%S")

        # Determine the result based on the rounded prediction
        if rounded_prediction == 0:
            result = {'Date': timestamp, 'prediction': float(predictions), 'prediction (%)': rounded_percentage, 'rounded prediction': 0, 'status': 'Default'}
        else:
            result = {'Date': timestamp, 'prediction': float(predictions), 'prediction (%)': rounded_percentage, 'rounded prediction': 1, 'status': 'Not Default'}

        return result

@app.post("/predict")
async def predict_loan(data: dict):
    loan_predictor = LoanPrediction(data)
    result = loan_predictor.predict_loan()
    return result

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)