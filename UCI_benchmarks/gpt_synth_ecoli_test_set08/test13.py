import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Separate the data into features and targets
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]  # All rows, only the last column
# Replace the categorical targets with numerical labels
classes = list(np.unique(y))
y = y.replace(classes, range(len(classes)))
# Convert the pandas DataFrame to numpy array
X = X.to_numpy()
y = y.to_numpy()
# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Standardize the training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Use RandomForestClassifier for the model
model = RandomForestClassifier(n_estimators=1000, random_state=42)
model.fit(X_train, y_train)
def predict_label(sample):
    # Preprocess the sample data and predict probabilities
    sample = np.array(sample).reshape(1, -1)
    sample = scaler.transform(sample)
    probabilities = model.predict_proba(sample)[0]
    # Return probabilities in the same format as the original model
    if len(probabilities) < len(classes):
        probabilities = np.append(probabilities, [0]*(len(classes) - len(probabilities)))
    return probabilities