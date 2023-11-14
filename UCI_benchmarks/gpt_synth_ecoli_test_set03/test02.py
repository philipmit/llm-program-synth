import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, all the columns except the last one 
y = ecoli.iloc[:, -1]  # All the rows, only the last
y = y.replace(np.unique(y), list(range(len(np.unique(y)))))
# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Initializing a scalar
scaler = StandardScaler()
# Fitting scalar on X_train
X_train = scaler.fit_transform(X_train)
# define the model
model = LogisticRegression(max_iter=1000, penalty='l2')
# fit the model
model.fit(X_train, y_train)
# Defining function that takes raw unprocessed data for a single sample
# function should return the predicted label for that sample
def predict_label(raw_data):
    ready_data = scaler.transform(raw_data.reshape(1, -1))  # Preprocess the data 
    probabilities = model.predict_proba(ready_data)  # Apply the model to make predictions
    return probabilities[0]  # Return the probabilities for the sample