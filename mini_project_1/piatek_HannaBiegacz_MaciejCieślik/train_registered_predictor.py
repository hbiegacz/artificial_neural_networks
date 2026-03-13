import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os


# LOAD DATASET, REMOVE UNNCESARY FEATURES
data_path = os.path.join('data', 'training_data.csv')
df = pd.read_csv(data_path)
target = 'registered'
features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
X = df[features]
y = df[target]

# PREPARE DATA -> NORMALIZE, TURN INTO NUMERICAL REPRESENTATION
# ADDING THIS STEP DEFINITELY MADE THE FINAL rmsle ERROR
cat_features = ['season', 'mnth', 'hr', 'weekday', 'weathersit']
num_features = ['temp', 'atemp', 'hum', 'windspeed']
bin_features = ['yr', 'holiday', 'workingday']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features), # converts into numerical representation
        ('bin', 'passthrough', bin_features)
    ])

X_processed = preprocessor.fit_transform(X)
if hasattr(X_processed, 'toarray'): 
    X_processed = X_processed.toarray()

# DIVIDE THE DATASET INTO TRAINIG & TESTING
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)



# WE EXPERIMENTALLY ADDED THE LAYERS & DROPOUTS, COMPARING THE RESULTS 
# AND THIS CONFIG (3 LAYERS) PROVED THE BEST
# WE THEN TRIED ADDING THE DROPOUTS TO FURTHER IMPROVE RESULTS
registeredPredictor = models.Sequential([
    layers.Input(shape=(X_processed.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='linear')
])

registeredPredictor.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = registeredPredictor.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[tf.keras.callbacks.EarlyStopping(min_delta=0.01, patience=10, restore_best_weights=True)]
)

# SAVE SO WE CAN USE OUR TRAINED MODEL TO MAKE PRREDICTIONS IN MAKE_PREDICTIONS.PY
registeredPredictor.save('models/registered_model.keras')
joblib.dump(preprocessor, 'preprocessor.joblib')