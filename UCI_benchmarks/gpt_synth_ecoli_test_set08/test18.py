import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1] 
y = ecoli.iloc[:, -1]
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
X = X.values
y = y.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize features
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
# Use an optimized logistic regression model for multi-class classification
model = LogisticRegressionCV(cv=5, random_state=0, multi_class='multinomial').fit(X_train, y_train)
def predict_label(single_sample):
    processed_sample = scaler.transform(np.array(single_sample).reshape(1, -1))
    predicted_probabilities = model.predict_proba(processed_sample)
    return predicted_probabilities[0]