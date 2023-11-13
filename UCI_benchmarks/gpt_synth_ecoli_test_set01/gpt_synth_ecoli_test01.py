import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Split the dataset into features X and labels y
X = ecoli.iloc[:, 1:-1]
y = ecoli.iloc[:, -1]
# Replace strings with numbers in y
classes = np.unique(y)
num_classes = len(classes)
y = y.replace(list(classes), list(range(num_classes)))
# Convert pandas DataFrames to numpy arrays
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train the logistic regression model
model = LogisticRegression(multi_class='ovr', max_iter=1000)
model.fit(X_train, y_train)
# Define the prediction function
def predict_label(raw_sample):
    # Reshape the raw sample if it has one dimension
    if len(raw_sample.shape) == 1:
        raw_sample = raw_sample.reshape(1, -1)
    # Standardize the raw sample
    raw_sample = scaler.transform(raw_sample)
    # Initialize a list to hold the predicted probabilities for all classes
    predicted_probs = [0] * num_classes
    # Predict the probabilities for the raw sample
    probs = model.predict_proba(raw_sample)[0]
    # Assign the predicted probabilities to the corresponding classes
    for i, prob in zip(model.classes_, probs):
        predicted_probs[i] = prob
    # Return the predicted probabilities
    return predicted_probs