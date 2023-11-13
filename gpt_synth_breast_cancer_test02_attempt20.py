from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
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
# Gradient Boosting Classifier tends to perform better compared to Random Forest
gb = GradientBoostingClassifier(random_state=42)
# Define a broader range of parameters for Grid Search
param_grid = [
    {'n_estimators': [100, 200, 300], 'max_depth' : [3, 5, 10], 'min_samples_split' : [2, 5, 10],
    'learning_rate': [0.01, 0.1, 1], 'subsample': [0.5, 0.75, 1]}
]
# Grid Search to find the best hyperparameters
clf = GridSearchCV(gb, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
clf.fit(X_train, y_train)
def predict_label(single_sample):
    sample = scaler.transform(np.reshape(single_sample, (1, -1)))  # Rescale and reshape data
    prob = clf.predict_proba(sample)[0][1]  # Get the probability of label 1
    return prob