import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
# Load the dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, all columns excluding sequence name and class
y = ecoli.iloc[:, -1]   # All rows, only the class column
# Replace string labels in y with numbers
lb = LabelBinarizer()
y = lb.fit_transform(y)  
y = [np.where(r==1)[0][0] for r in y]  # convert y from 2D array to 1D
X = X.to_numpy()
y = np.array(y)
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80-20 split for more training data
# Standardize train features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)  
# Initialising GradientBoostingClassifier for higher performance
model = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.02, max_depth=3, random_state=42)
model.fit(X_train_sc, y_train)
# Define the predict_label function
def predict_label(raw_input_sample):
    # Ensure raw_input_sample is numpy array and reshape(-1, 1) if necessary
    if isinstance(raw_input_sample, list):
        raw_input_sample = np.array(raw_input_sample)
    elif len(raw_input_sample.shape) == 1:
        raw_input_sample = raw_input_sample.reshape(1, -1)
    # Scale raw input sample
    sample = scaler.transform(raw_input_sample)
    # Perform prediction and return the probabilities for each class
    predicted_proba = model.predict_proba(sample)
    # Create a full length probabilities list for classes not predicted by the model
    full_length_proba = [0]*len(lb.classes_)
    for k, v in zip(model.classes_, predicted_proba[0]):
        full_length_proba[k] = v
    return full_length_proba