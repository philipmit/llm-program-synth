import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
# Load the ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', 
                    delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1].values    # All rows, all columns except the last one
y = ecoli.iloc[:, -1].values      # All rows, only the last column
# Preprocess y using Label Encoder
le = LabelEncoder()
y = le.fit_transform(y)
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Train RandomForestClassifier instead of LogisticRegression to enhance predictive power
rf_clf = RandomForestClassifier(n_estimators=200, max_depth=None, 
                                random_state=0)
# Fit
rf_clf.fit(X_train, y_train)
# Predict
def predict_label(sample):
    sample = np.array(sample).reshape(1, -1)
    # Scale the sample
    sample = scaler.transform(sample)
    predicted_probabilities = rf_clf.predict_proba(sample)
    # Create an array to hold probabilities for each class
    probabilities = np.zeros(len(le.classes_))
    # Assign predicted probabilities to respective index
    probabilities[rf_clf.classes_] = predicted_probabilities[0]
    return probabilities