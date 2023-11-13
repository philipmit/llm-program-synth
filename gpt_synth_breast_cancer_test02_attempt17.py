from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
# Standardize the dataset for better performance in Logistic Regression
scaler = StandardScaler()
X_standard = scaler.fit_transform(X)
# Split the standardized dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_standard, y, test_size=0.5, random_state=42)
# Set model parameters for GridSearch
params = {'C': np.logspace(-3, 3, 7), 
          'penalty': ['l1', 'l2'],
          'solver': ['liblinear', 'saga']}
# Initialize Logistic Regression Model
model = LogisticRegression()
# Grid search of parameters using cross-validation, fitting to the training data
grid_model = GridSearchCV(model, params, cv=5, scoring='roc_auc')
grid_model.fit(X_train, y_train)
# Function takes raw data for a single sample and returns predicted probability of class=1
def predict_label(raw_data):
    # Standardize the raw_data similar to the training data
    raw_data_std = scaler.transform(np.array(raw_data).reshape(1, -1))
    # predict_proba returns a 2D array with probabilities of class=0 at 0th index and class=1 at 1st index
    return grid_model.predict_proba(raw_data_std)[0][1]