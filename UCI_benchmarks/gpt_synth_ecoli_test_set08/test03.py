import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# Load the dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Prepare the feature and target variables
X = ecoli.iloc[:, 1:-1]
y = ecoli.iloc[:, -1]
y = y.replace(list(np.unique(y)), range(len(np.unique(y))))
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Train the logistic regression model
log_reg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
def predict_label(raw_data_sample):
    return log_reg.predict_proba(raw_data_sample.reshape(1, -1))