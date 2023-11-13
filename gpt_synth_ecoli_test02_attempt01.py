import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Get features X and label y
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# Replace strings with integer labels in y
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# Split the dataset into training and testing sets using stratify to balance the classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train the Logistic Regression model with multinomial setting for multi-class
lrModel = LogisticRegression(random_state=42, max_iter=5000, multi_class='multinomial')
lrModel.fit(X_train, y_train)
# Define the function predict_label to return probabilities for all classes
def predict_label(raw_data):
    raw_data = np.array(raw_data)
    raw_data_scaled = scaler.transform(raw_data.reshape(1, -1))
    prediction_proba = lrModel.predict_proba(raw_data_scaled)[0]
    return prediction_proba.tolist()