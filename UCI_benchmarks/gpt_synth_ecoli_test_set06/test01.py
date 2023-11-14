import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
def preprocess_data(raw_data):
    raw_data = raw_data.reshape(1, -1)
    raw_data = scaler.transform(raw_data)
    return raw_data
def predict_label(raw_data):
    processed_data = preprocess_data(raw_data)
    probabilities = np.zeros(num_classes)
    prediction = model.predict_proba(processed_data)
    for i, val in enumerate(model.classes_):
        probabilities[val] = prediction[0][i]
    return probabilities
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1] 
y = ecoli.iloc[:, -1]
label_unique = list(np.unique(y))
num_classes = len(label_unique)
y = y.replace(label_unique, list(range(num_classes)))
X = X.to_numpy()
y = y.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Initialize Scaler
scaler = StandardScaler()
# Fit and transform the training data
X_train = scaler.fit_transform(X_train)
# Initialize a RandomForest model
model = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=0)
# Fit the model
model.fit(X_train, y_train)