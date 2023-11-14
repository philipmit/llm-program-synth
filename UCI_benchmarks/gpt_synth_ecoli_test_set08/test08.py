import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
# Load dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1].values  # All rows, all columns except last one
y = ecoli.iloc[:, -1].values   # All rows, only the last column
# Use LabelEncoder to encode class labels as numbers
le = LabelEncoder()
y = le.fit_transform(y)
# Ensure stratified split
stratified_split = StratifiedKFold(n_splits=2, random_state=42, shuffle=True)
train_index, test_index = next(stratified_split.split(X, y))
# Split the dataset
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]
# Train logistic regression
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
logreg.fit(X_train, y_train)
def predict_label(X):
    # Assume the input x is a list with 7 elements
    # Convert it to (1, 7) shape numpy.ndarray
    X = np.array(X).reshape(1, -1)
    # Get probabilities as output of the logistic regression model
    prob = logreg.predict_proba(X)[0]
    # Check if all classes have their probabilities
    missing_classes = set(le.classes_) - set(logreg.classes_)
    for c in missing_classes:
        prob = np.insert(prob, le.transform([c])[0], 0)
    return prob