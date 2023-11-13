import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  
y = ecoli.iloc[:, -1]   
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7])
X = X.to_numpy()
y = y.to_numpy()
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
    return logreg.predict_proba(sample)