import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import RandomForestClassifier
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1] 
y = ecoli.iloc[:, -1]  
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7])
X = X.to_numpy()
y = y.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Scale the datasets using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Training a Random Forest Classifier
model = RandomForestClassifier(n_estimators=500, max_depth=8, random_state=42)
model.fit(X_train, y_train)
def predict_label(X_raw):
    num_class = len(np.unique(y))  
    X_raw_np_array = np.array(X_raw, dtype=float).reshape(1, -1)
    X_raw_scaled = scaler.transform(X_raw_np_array)
    prediction = model.predict_proba(X_raw_scaled)[0]
    if len(prediction) != num_class:
        diff = num_class - len(prediction)  
        return np.concatenate([prediction, np.zeros(diff)])  
    else:
        return prediction