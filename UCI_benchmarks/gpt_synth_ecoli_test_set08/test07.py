import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
def predict_label(sample):
    # Reshape and 
    sample = np.array(sample).reshape(1, -1)
    sample = scaler.transform(sample)
    # Predict probabilities
    predicted_probs = model.predict_proba(sample)
    return predicted_probs[0]
# Load the dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Create feature matrix X and target vector y
X = ecoli.iloc[:, 1:-1]
y = ecoli.iloc[:, -1]
# Replace strings with numbers in y
unique_y = np.unique(y)
y = y.replace(unique_y, np.arange(len(unique_y)))
X = X.to_numpy()
y = y.to_numpy()
# Use StratifiedShuffleSplit to hold out test set with all classes represented
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
# Apply scaling on X
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
# Train logistic regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)