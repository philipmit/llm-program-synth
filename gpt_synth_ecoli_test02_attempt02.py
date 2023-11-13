import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Load the digits dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# replace strings with numbers in y
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7])
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
logisticRegr = LogisticRegression(max_iter=1000)
logisticRegr.fit(X_train, y_train)
def predict_label(X):
    X = X[1:-1]  # Exclude 'Sequence Name' and 'class'
    X = X.astype(float)  # Convert to float as the model has been trained on float data
    X = X.reshape(1, -1)  # Reshape since the model expects a 2D array
    return logisticRegr.predict_proba(X)