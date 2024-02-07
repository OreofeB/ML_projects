import chardet
import numpy as np
import pickle
import pandas as pd 
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.utils import resample


def preprocess_data(data_path):
    # Load and preprocess data
    with open(data_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['utf-8']

    data = pd.read_csv(data_path, encoding=encoding)

    columns_to_drop = ['org_id', 'user_id', 'loan_id', 'work_start_date', 'work_email', 'loan_request_day',
                       'current_employer', 'work_email_validated', 'first_account', 'last_account', 'created_on',
                       'process_time', 'photo_url', 'logins']

    new_data = data.drop(columns=columns_to_drop)
    new_data = new_data.dropna()
    new_data = new_data[new_data['status_id'] != 1]

    new_data["requested_amount"] = new_data["requested_amount"].astype(int)

    columns_to_encode = ['gender', 'marital_status', 'type_of_residence', 'educational_attainment',
                         'sector_of_employment', 'monthly_net_income', 'country', 'city', 'lga', 'purpose',
                         'selfie_bvn_check', 'selfie_id_check', 'device_name', 'mobile_os', 'os_version',
                         'no_of_dependent', 'employment_status', 'status_id']

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

    df_majority_paid = new_data[new_data["status_id"] == 0]
    df_minority_default = new_data[new_data["status_id"] == 1]

    df_majority_downsampled = resample(df_majority_paid,
                                       replace=False,
                                       n_samples=len(df_minority_default),
                                       random_state=123)
    
    df_minority_upsampled = resample(df_minority_default,
                                       replace=True,
                                       n_samples=len(df_majority_paid),
                                       random_state=123)

    us_data = pd.concat([df_majority_paid, df_minority_upsampled])

    ds_data = pd.concat([df_majority_downsampled, df_minority_default])

    # ds_y = ds_data['status_id']
    # ds_X = ds_data.drop(columns='status_id')

    # us_y = us_data['status_id']
    # us_X = us_data.drop(columns='status_id')

    # y = new_data['status_id']
    # X = new_data.drop(columns='status_id')

    return ds_data, us_data


# Define the path to your dataset
data_path = 'C:\\Users\\parkway\\Documents\\Python Code\\Dissertation_code\\data.csv'

# Preprocess the data
ds_data, us_data = preprocess_data(data_path)

# Split the data into training and testing sets
train_data, test_data = train_test_split(us_data, train_size=0.7)

# Define target and feature variables
y = train_data['status_id']
X = train_data.drop(columns='status_id')
X_test = test_data.drop(columns='status_id')
y_test = test_data['status_id']

# Split the training data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=4)

# Standardize the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_valid = sc.transform(X_valid)
X_test = sc.transform(X_test)

print('train shape:', train_data.shape, 'test shape:', test_data.shape)
print('X_train shape:', X_train.shape, 'X_valid shape:', X_valid.shape)

# Build the neural network model
model = Sequential()
model.add(Dense(units=128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.2, seed=123))
model.add(Dense(units=64, activation='tanh'))
model.add(BatchNormalization())
model.add(Dropout(0.2, seed=123))
model.add(Dense(units=32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2, seed=123))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# Define callbacks for training
callbacks = [
    EarlyStopping(monitor="accuracy", patience=5, verbose=1),
    ReduceLROnPlateau(factor=0.15, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('dp_model.h5', verbose=1, save_best_only=True, save_weights_only=False)
]

# Train the model
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=50, epochs=500, callbacks=callbacks)

pickle.dump(model, open('dp_model.pkl', 'wb'))

# Load the trained model
model = tf.keras.models.load_model('dp_model.h5')

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Generate confusion matrix and metrics
cm = confusion_matrix(y_test, y_pred.round())
precision = precision_score(y_test, y_pred.round())
recall = recall_score(y_test, y_pred.round())
accuracy = accuracy_score(y_test, y_pred.round())
print('Precision:', precision * 100)
print('Recall:', recall * 100)
print('Accuracy:', accuracy * 100)
print(cm)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm, cmap='Blues')
ax.grid(False)
ax.set_xlabel('Predicted labels', fontsize=12, color='black')
ax.set_ylabel('True labels', fontsize=12, color='black')
ax.set_xticks(range(2))
ax.set_yticks(range(2))
ax.set_xticklabels(['Paid', 'Not Paid'], fontsize=10, color='black')
ax.set_yticklabels(['Paid', 'Not Paid'], fontsize=10, color='black')
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)

# Add annotations
thresh = cm.max() / 2
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', color='black' if cm[i, j] > thresh else 'black')

plt.show()





