import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Function to preprocess raw data
def preprocess_data(raw_data):
    # Standardize
    raw_data = scaler.transform(raw_data)
    return raw_data
# Function to predict label for raw data
def predict_label(raw_data):
    # Raw data is reshaped since the input is expected to be 2D
    raw_data = raw_data.reshape(1, -1)
    # Preprocess the raw data
    processed_data = preprocess_data(raw_data)
    # Predict the probabilities using trained model
    probabilities = model.predict_proba(processed_data)
    # Return the class with highest probability
    return probabilities.flatten()
# Load the dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1] 
y = ecoli.iloc[:, -1]   
# Replace strings with numbers in y
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7])
# Convert DataFrame to np.array
X = X.to_numpy()
y = y.to_numpy()
# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train the logistic regression model
model = LogisticRegression(multi_class='multinomial')
model.fit(X_train, y_train)
scaler.fit(X_train)