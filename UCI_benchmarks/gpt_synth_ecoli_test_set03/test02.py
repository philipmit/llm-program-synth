import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, all the columns except the last one 
y = ecoli.iloc[:, -1]  # All the rows, only the last
# Transform string values to numerical
le = LabelEncoder()
y = le.fit_transform(y)
# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Initializing a scalar and model
scaler = StandardScaler()
model = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.01, random_state=42) # Changed model to GradientBoosting 
# Fitting the scalar and model on X_train
X_train = scaler.fit_transform(X_train)
model.fit(X_train, y_train)
# Defining function that takes raw unprocessed data for a single sample
def predict_icu_mortality(raw_data):
    processed_data = scaler.transform(raw_data.reshape(1, -1))  # Preprocess the data 
    probabilities = model.predict_proba(processed_data)  # Apply the model to make predictions
    # Create a probability list by padding zeros for absent classes to align length with 8 (Number of classes)
    p = probabilities[0].tolist() + [0]*(8 - len(probabilities[0]))
    return p