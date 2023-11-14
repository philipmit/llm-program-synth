import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
# Load the dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Prepare the feature and target variables
X = ecoli.iloc[:, 1:-1]
y = ecoli.iloc[:, -1]
# Transform target labels into binary array
lb = LabelBinarizer()
y = lb.fit_transform(y)
# If y has only 1 column, i.e., only 2 unique labels, reshape it to (n_samples,)
if y.shape[1] == 1:
    y = y.reshape(-1)
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Train the logistic regression model
log_reg = LogisticRegression(max_iter=1000)
if len(y_train.shape) > 1 and y_train.shape[1] > 1:
    # If y_train has more than 1 column, i.e., more than 2 unique labels
    from sklearn.multiclass import OneVsRestClassifier
    cls = OneVsRestClassifier(log_reg)
else:
    cls = log_reg
cls.fit(X_train, y_train)
def predict_label(raw_data_sample):
    proba = cls.predict_proba(raw_data_sample.reshape(1, -1))
    return proba[0]