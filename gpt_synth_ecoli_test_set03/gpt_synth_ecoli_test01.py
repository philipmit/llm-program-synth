import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Load the dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
# Assign column names
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Assign predictor variables
X = ecoli.iloc[:, 1:-1]  
# Assign target variable
y = ecoli.iloc[:, -1]   
# Replace classes by numbers
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7])
# Convert pandas dataframe to numpy arrays
X = X.to_numpy()
y = y.to_numpy()
# Split dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Normalize the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train logistic regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
# Define the function predict_label
def predict_label(sample):
    sample = scaler.transform(np.array(sample).reshape(1, -1))
    return np.argmax(logreg.predict_proba(sample))