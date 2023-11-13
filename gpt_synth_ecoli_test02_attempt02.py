import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
# Load the dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# Replace strings with numbers in y
y_encoded = pd.get_dummies(y)
X = X.to_numpy()
y = y_encoded.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Train Logistic Regression Model
logisticRegr = LogisticRegression(multi_class='multinomial', max_iter=1000)
logisticRegr.fit(X_train, y_train)
# Define a prediction function
def predict_label(X):
    if isinstance(X, list):
        X = np.array(X)
    X = X.astype(float)  # Convert numbers to float as the model has been trained on float data
    X = X.reshape(1, -1)  # Reshape to meet the requirement of the model's predict_proba() function
    return logisticRegr.predict_proba(X)[0]  # It's crucial to only return the first element of the output as it's a 2D array