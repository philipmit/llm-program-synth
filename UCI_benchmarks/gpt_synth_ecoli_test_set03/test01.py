import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
# Load the dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Define features and target
X = ecoli.drop(['Sequence Name', 'class'], axis=1)
y = ecoli['class']
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7])
# Cast to numpy arrays
X = X.to_numpy()
y = y.to_numpy()
# Stratified Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) # reduced test size to 0.3 to have more data for training.
# Pre-processing: Standard Scaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Train RandomForestClassifier with increased estimators for better accuracy
rfc = RandomForestClassifier(n_estimators=2000, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', bootstrap=True, oob_score=False, n_jobs=-1, random_state=42, verbose=0, warm_start=False, class_weight=None)
rfc.fit(X_train, y_train)
def predict_icu_mortality(data):
    data = sc.transform(np.array(data).reshape(1, -1)) # apply the standard scalar on new data
    proba = rfc.predict_proba(data)[0]
    return proba