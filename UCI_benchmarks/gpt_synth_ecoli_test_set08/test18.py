from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
# Load the digits dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# replace strings with numbers in y
y = y.replace(list(np.unique(y)), list(range(len(np.unique(y)))))
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Logistic regression
lr = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter = 4000)
lr.fit(X_train, y_train)
def predict_label(single_sample):
    processed_sample = scaler.transform(np.array(single_sample).reshape(1, -1))
    predicted_probabilities = lr.predict_proba(processed_sample)
    return predicted_probabilities[0]