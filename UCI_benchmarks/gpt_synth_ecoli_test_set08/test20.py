import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
# Load the digits dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1].values # All rows, all columns except the last one
y = ecoli.iloc[:, -1].values # All rows, only the last column
# Label Encoding for multi-class classification
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Implement RandomForestClassifier to potentially increase AUC
rf_clf = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42)
# Model Training
rf_clf.fit(X_train, y_train)
# Predict function
def predict_label(sample):
    sample = np.array(sample).reshape(1, -1)
    sample = sc.transform(sample) # scaling
    pred_proba = rf_clf.predict_proba(sample) # Prediction
    probabilities = np.zeros(len(label_encoder.classes_))
    probabilities[rf_clf.classes_] = pred_proba[0]
    return probabilities