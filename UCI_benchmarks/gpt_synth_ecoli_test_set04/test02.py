import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# replace strings with numbers in y
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7])
X = X.to_numpy()
y = y.to_numpy()
# Split the Ecoli dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Fit a logistic regression model on the training dataset
log_reg = LogisticRegression(multi_class='ovr',solver='liblinear')
log_reg.fit(X_train, y_train)
def predict_label(raw_sample):
    # Scale the raw_sample and reshape it to 2D
    raw_sample = raw_sample.reshape(1, -1)
    sample = scaler.transform(raw_sample)
    # Predict the probabilities
    predicted_proba = log_reg.predict_proba(sample)
    return predicted_proba[0]