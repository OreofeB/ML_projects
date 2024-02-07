import keras
import chardet
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.applications import InceptionV3, NASNetMobile, EfficientNetB7, EfficientNetV2B1
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, recall_score
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

def preprocess_data(data_path):
    # Load and preprocess data
    with open(data_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']

    data = pd.read_csv(data_path, encoding=encoding)

    # Your existing preprocessing code from "code 1"
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
    
    ds_data = pd.concat([df_majority_downsampled, df_minority_default])
    
    
    df_minority_upsampled = resample(df_minority_default,
                                       replace=True,
                                       n_samples=len(df_majority_paid),
                                       random_state=123)
    
    us_data = pd.concat([df_majority_paid, df_minority_upsampled])


    return us_data, ds_data


data_path = 'C:\\Users\\parkway\\Documents\\Dissertation\\data.csv'
us_data, ds_data = preprocess_data(data_path)
new_data = us_data
train_data, test_data = train_test_split(new_data, train_size=0.7)

# y = train_data['status_id']
# X = train_data.drop(columns='status_id')

# Calculate the size of the square matrix

matrix_size = 128
ids = len(test_data)

X = np.ones((ids, matrix_size, matrix_size,1), dtype=float)  # Initialize X with ones
y = np.zeros(ids, dtype=float)

counter = 0
for index, row in test_data.iterrows():
    x_values = row.drop('status_id').values
    x_center = np.array(x_values)

    x_matrix = np.ones((matrix_size, matrix_size))  # Initialize matrix with ones
    center_row = (matrix_size - x_center.shape[0]) // 2
    center_col = (matrix_size - x_center.shape[0]) // 2
    x_matrix[center_row:center_row + x_center.shape[0],
    center_col:center_col + x_center.shape[0]] = x_center.reshape(-1, 1)

    X[counter] = x_matrix.reshape(matrix_size, matrix_size, 1) / 255  # Normalize

    y[counter] = row['status_id']
    counter += 1

############ 

ids = len(train_data)

X_test = np.ones((ids, matrix_size, matrix_size,1), dtype=float)  # Initialize X with ones
y_test = np.zeros(ids, dtype=float)

counter = 0
for index, row in train_data.iterrows():
    x_values = row.drop('status_id').values
    x_center = np.array(x_values)

    x_matrix = np.ones((matrix_size, matrix_size))  # Initialize matrix with ones
    center_row = (matrix_size - x_center.shape[0]) // 2
    center_col = (matrix_size - x_center.shape[0]) // 2
    x_matrix[center_row:center_row + x_center.shape[0],
    center_col:center_col + x_center.shape[0]] = x_center.reshape(-1, 1)

    X_test[counter] = x_matrix.reshape(matrix_size, matrix_size, 1) / 255  # Normalize

    y_test[counter] = row['status_id']
    counter += 1


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.30, random_state=2019)


# from sklearn.preprocessing import StandardScaler

# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_valid = sc.transform(X_valid)

print('train shape:',train_data.shape,'test shape:', test_data.shape)

print('X_train shape:', X_train.shape, 'X_valid shape:', X_valid.shape)


# Step 2: Build and compile the model
input_tensor = Input(shape=(matrix_size, matrix_size, 3))
# base_model = NASNetMobile(include_top=False, weights='imagenet')
# base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)
# base_model = EfficientNetB7(input_tensor=input_tensor, weights='imagenet', include_top=False)
base_model = EfficientNetV2B1(input_tensor=input_tensor, include_top=False, weights="imagenet")
base_model.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
x = Dense(32, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
predictions = Dense(1, activation='sigmoid')(x)  # Assuming binary classification
model = keras.Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["binary_accuracy"])
model.summary()


# Step 3: Define callbacks for training
callbacks = [
    EarlyStopping(monitor="binary_accuracy", patience=5, verbose=1),
    ReduceLROnPlateau(factor=0.15, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint('cnn_model_pt.h5', verbose=1, save_best_only=True, save_weights_only=False)
]

# Step 4: Train the model using the cleaned and preprocessed data
model.fit(X_train, y_train, validation_data=(X_valid,y_valid), batch_size=50, epochs=100, callbacks=callbacks)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('cnn_model_pt.h5')

# Make predictions on the testing data
y_pred = model.predict(X_test)

# generate confusion matrix
cm = confusion_matrix(y_test, y_pred.round())
precision = precision_score(y_test, y_pred.round())
recall = recall_score(y_test, y_pred.round())
accuracy = accuracy_score(y_test, y_pred.round())
print('Precision:',precision * 100)
print('Recall:',recall * 100)
print('Accuracy:', accuracy * 100)
print(cm)

# plot confusion matrix
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

# add annotations
thresh = cm.max() / 2
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', color='black' if cm[i, j] > thresh else 'black')

plt.show()
