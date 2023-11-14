import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
# Load the dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# Label binarize the string labels in y
lb = LabelBinarizer()
y = lb.fit_transform(y)
# converting the 2-D array to 1-D
y = [np.where(r==1)[0][0] for r in y]
X = X.to_numpy()
y = np.array(y)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train the RandomForestClassifier model
model = RandomForestClassifier(n_estimators=1000, max_depth=None, random_state=42)
model.fit(X_train, y_train)
# Define a predict function
def predict_label(raw_input_sample):
    # Ensure the raw_input_sample is numpy array and reshape(-1, 1) if necessary
    if isinstance(raw_input_sample, list):
        raw_input_sample = np.array(raw_input_sample)
    elif len(raw_input_sample.shape) == 1:
        raw_input_sample = raw_input_sample.reshape(1, -1)
    # Normalize the raw input sample
    sample = scaler.transform(raw_input_sample)
    # Perform prediction and return the probability for each class
    probabilities = model.predict_proba(sample)
    # Creating a full length probabilities list for classes not predicted by the model
    full_length_proba = [0]*len(lb.classes_)
    for k, v in zip(model.classes_, probabilities[0]):
        full_length_proba[k] = v
    return full_length_proba