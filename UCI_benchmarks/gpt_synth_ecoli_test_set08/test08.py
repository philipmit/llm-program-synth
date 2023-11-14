import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Load dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# replace strings with numbers in y
y = y.replace(np.unique(y), np.arange(len(np.unique(y))))
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Train logistic regression
logreg = LogisticRegression(multi_class='multinomial',solver='lbfgs')
logreg.fit(X_train, y_train)
def predict_label(x):
    # Assume the input x is a list with 7 elements
    # Convert it to (1, 7) shape numpy.ndarray
    x = np.array(x).reshape(1, -1)
    probs = logreg.predict_proba(x)
    return probs[0]