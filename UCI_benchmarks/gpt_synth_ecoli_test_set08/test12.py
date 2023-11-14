import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Load the dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# Replace strings with numbers in y
np.unique( y)
len(list(np.unique( y)))
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7])
X=X.to_numpy()
y=y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train the logistic regression model
model = LogisticRegression(multi_class='ovr', solver='liblinear')
model.fit(X_train, y_train)
# Define a predict function
def predict_label(raw_input_sample):
    # Ensure the raw_input_sample is numpy array and reshape(-1, 1) if necessary
    if isinstance(raw_input_sample, list):
        raw_input_sample = np.array(raw_input_sample)
    if len(raw_input_sample.shape) == 1:
        raw_input_sample = raw_input_sample.reshape(1, -1)
    # Normalize the raw input sample
    sample = scaler.transform(raw_input_sample)
    # Perform prediction and return the probability for each class
    probabilities = model.predict_proba(sample)
    return probabilities[0]  # flatten the result