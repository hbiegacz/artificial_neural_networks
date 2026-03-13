import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# LOAD TRAINED PREDICTORS
casualPredictor = tf.keras.models.load_model('casual_model.keras')
registeredPredictor = tf.keras.models.load_model('registered_model.keras')

# LOAD & PREPARE DATASET (REMOVE DATE, etc)
preprocessor = joblib.load('preprocessor.joblib')
eval_df = pd.read_csv('evaluation_data.csv')
features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
X_eval = eval_df[features]

X_eval_processed = preprocessor.transform(X_eval)
if hasattr(X_eval_processed, 'toarray'):
    X_eval_processed = X_eval_processed.toarray()

# USE BOTH MODELS TO CREATE PREDICTIONS
registered_predictions = casualPredictor.predict(X_eval_processed)
registered_predictions = np.maximum(0, registered_predictions)
registered_predictions = np.round(registered_predictions, 16)

casual_predictions = casualPredictor.predict(X_eval_processed)
casual_predictions = np.maximum(0, casual_predictions)
casual_predictions = np.round(casual_predictions, 16)

# SUM THE PREDICTIONS TO RECEIVE THE FINAL 'CNT' VALUE
final_predictions = registered_predictions + casual_predictions
np.savetxt('results/predictions.csv', final_predictions, delimiter=',', fmt='%.10f')