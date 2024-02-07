import pickle
import h5py
import tensorflow as tf 

# Replace 'your_model.h5' with the path to your HDF5 model file
hdf5_model_path = 'C:\\Users\\parkway\\Documents\\Python Code\\LP_Models\\LP_Models\\models\\cnn_model_pt.h5'

# Load the HDF5 model
loaded_model = tf.keras.models.load_model(hdf5_model_path)

# Replace 'your_model.pkl' with the desired path for the pickle file
pickle_model_path = 'C:\\Users\\parkway\\Documents\\Python Code\\LP_Models\\LP_Models\\models\\cnn_model_pt.pkl'

# Save the loaded model as a pickle file
with open(pickle_model_path, 'wb') as pickle_file:
    pickle.dump(loaded_model, pickle_file)
