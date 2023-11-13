import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
# Load dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# replace strings with numbers in y
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # This will assign a unique number to each class label starting from zero
# Convert DataFrames to numpy arrays
X = X.values
y = y.astype('int')
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Create and train logistic regression model
# Setting multi_class='ovr' makes the logistic regression a one-vs-rest classifier which is suitable for multi-class problems.
lr = LogisticRegression(max_iter=10000, multi_class='ovr')
lr.fit(X_train, y_train)
# Function to predict labels for a given sample
def predict_label(raw_sample):
    processed_sample = np.array([raw_sample])  # Process the raw unprocessed sample to match the format of X
    num_classes = len(np.unique(y))  # Get the total number of classes
    proba = lr.predict_proba(processed_sample)[0]  # Predict probabilities
    proba.resize(num_classes)  # Resize the array in case a class was not in the training set and its probability is missing
    return proba  # Return the probabilities for each class