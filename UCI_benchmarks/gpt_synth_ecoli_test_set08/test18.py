import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Prepare predictors and target variables
X = ecoli.iloc[:, 1:-1].copy() 
y = ecoli.iloc[:, -1].copy()
# Use label encoder to transform textual classes/labels to numerical
le = LabelEncoder()
y = le.fit_transform(y)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Scale features
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
# Train a logistic regression model
clf = LogisticRegressionCV(cv=5, random_state=0, multi_class='multinomial').fit(X_train_scaled, y_train)
# The predict_label function
def predict_label(raw_sample):
    # Preprocessing the single raw sample
    sample = np.array(raw_sample).reshape(1, -1)
    processed_sample = scaler.transform(sample)
    # Use clf to predict the sample's probabilities
    predicted_probabilities = clf.predict_proba(processed_sample)
    # Return an array of zeroes for classes not present in the training set
    full_pred = np.zeros(len(le.classes_))
    full_pred[:predicted_probabilities.shape[1]] = predicted_probabilities[0]
    return full_pred