import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
# Load the dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Split dataset into features and target
X = ecoli.iloc[:, 1:-1]  # Features: All rows, all columns excluding the first and last one
y = ecoli.iloc[:, -1]    # Target: All rows, only the last column
# Convert string labels into numerical labels
le = LabelEncoder()
y = le.fit_transform(y)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train Logistic Regression Model
model = LogisticRegression(multi_class='ovr', max_iter=1000)
model.fit(X_train, y_train)
# Define the predict_label function
def predict_label(sample):
    sample = scaler.transform([sample])  # Apply standard scaling to the test sample
    probabilities = model.predict_proba(sample)  # Get the predicted probabilities
    return probabilities[0]