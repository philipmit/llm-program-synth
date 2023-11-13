from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
# Feature scaling for improved performance
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Create a Random Forest Classifier and parameters for Grid Search
rf = RandomForestClassifier(random_state=42)
param_grid = [
    {'n_estimators': [50, 100, 200], 'max_depth' : [5, 10, 15], 'min_samples_split' : [2, 5, 10]}
]
# Grid Search to find the best hyperparameters
clf = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc')
clf.fit(X_train, y_train)
def predict_label(single_sample):
    sample = scaler.transform(np.reshape(single_sample, (1, -1)))  # Rescale and reshape data
    prob = clf.predict_proba(sample)[0][1]  # Get the probability of label 1
    return prob