import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
# Load the ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Get features and labels
X = ecoli.iloc[:, 1:-1]
y = ecoli.iloc[:, -1]
# replace strings with numbers in y
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# Standardize the data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Initialize and train RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
# Define the predict_label function
def predict_label(sample):
    sample_reshaped = np.array(sample).reshape(1, -1)
    # Standardize the test sample
    sample_reshaped = scaler.transform(sample_reshaped)
    predicted_probabilities = rf_model.predict_proba(sample_reshaped)
    # Add zero-probabilities for classes that were not in the training data
    complete_probabilities = np.zeros(len(label_encoder.classes_))
    for class_index, probability in enumerate(predicted_probabilities[0]):
        complete_probabilities[class_index] = probability
    return complete_probabilities