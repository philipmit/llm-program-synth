import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# Load the ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# Replace strings with numbers in y
y = y.replace(list(np.unique(y)), [0, 1, 2, 3, 4, 5, 6, 7])
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Feature scaling for normalization
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
# Train the model with one-vs-rest strategy
model = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)
# Define the predict_label function
def predict_label(sample):
    global scaler, model
    sample = np.array(sample).reshape(1, -1)
    sample = scaler.transform(sample)
    prediction = model.predict_proba(sample)
    return prediction.flatten()