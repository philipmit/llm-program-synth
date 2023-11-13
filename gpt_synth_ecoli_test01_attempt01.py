import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
# Naming the columns
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# replace strings with numbers in y
uniques = np.unique(y)
y = y.replace(uniques.tolist(), list(range(len(uniques))))
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Learning logistic regression model
lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train)
def predict_label(unprocessed_sample):
    processed_sample = np.array([unprocessed_sample[1:-1]])  # Process the raw unprocessed sample to match the format of X
    probabilities = lr.predict_proba(processed_sample)
    return probabilities