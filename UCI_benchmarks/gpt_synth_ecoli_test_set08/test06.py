import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np
# Load the ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Pre-process the dataset
X = ecoli.iloc[:, 1:-1]  # All rows, all the columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# Correct way to convert string labels into numbers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# Define prediction function
def predict_label(X_sample):
    X_sample = np.array(X_sample).reshape(1, -1)
    pred_proba = model.predict_proba(X_sample)
    return pred_proba[0]