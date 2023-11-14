# sklearn's logistic regression model and standard scaler are required, they should be imported
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Specify file path and load the Ecoli Dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Independant Variables
X = ecoli.iloc[:, 1:-1]  
# Dependant Variable
y = ecoli.iloc[:, -1]
# Replace class labels in y with numbers 
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7])
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize Data
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(X_train)
# Apply transform to both the training set and the test set.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# Initialize Logistic Regression Model
model = LogisticRegression(multi_class='ovr', max_iter=200)
# Fit Model
model.fit(X_train, y_train)
# Define the function for predicting labels
def predict_label(raw_sample):
    # reshaping the sample
    raw_sample = raw_sample.reshape(1, -1)
    # standardize the raw sample
    raw_sample = scaler.transform(raw_sample)
    # predict the probabilities for each class for the sample
    probability = model.predict_proba(raw_sample)
    return probability.ravel()  # ravel() is used to convert 2D array into 1D