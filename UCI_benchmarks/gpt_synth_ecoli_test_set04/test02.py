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
# Define the scaler
scaler = StandardScaler()
# Scale the X_train data
X_train = scaler.fit_transform(X_train)
# Fit a logistic regression model
log_reg = LogisticRegression(multi_class='ovr',solver='liblinear')
log_reg.fit(X_train, y_train)
def predict_label(raw_sample):
    # reshape raw_sample if it's only 1D (a single sample)
    if len(raw_sample.shape) == 1:
        raw_sample = raw_sample.reshape(1, -1)
    # normalize the sample
    sample = scaler.transform(raw_sample)
    # use the fitted model to predict_proba
    probas = log_reg.predict_proba(sample)
    # return the label with max probability
    return np.argmax(probas, axis=1)