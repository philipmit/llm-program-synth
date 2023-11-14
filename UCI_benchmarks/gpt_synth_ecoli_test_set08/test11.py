import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# Load the ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', 
                    delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows and all columns except the last one
y = ecoli.iloc[:, -1]   # All rows and only the last column
# Replace categorical classes with numerical labels in y
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7])
# Convert pandas DataFrame to numpy array
X = X.values
y = y.values
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize the features to have mean=0 and variance=1
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
# Define the number of classes
num_classes = len(np.unique(y))
# Instead of logistic regression let's try using a Random Forest Classifier which usually performs better
pipeline = make_pipeline(RandomForestClassifier())
param_grid = {
    'randomforestclassifier__n_estimators': [100, 200],
    'randomforestclassifier__max_depth': [5, 10, 15],
    'randomforestclassifier__max_features': ['auto', 'sqrt', 'log2'],
    'randomforestclassifier__criterion' :['gini', 'entropy']
}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc_ovr', n_jobs=-1)
grid.fit(X_train, y_train)
model = grid.best_estimator_
def predict_label(raw_data):
    # Preprocess the raw data
    raw_data = np.array(raw_data)
    raw_data = scaler.transform(raw_data.reshape(1, -1))
    # Predict the probabilities
    predicted_probabilities = model.predict_proba(raw_data)
    # Initialize a probabilities vector filled with zeros for each class type
    probabilities = np.zeros(num_classes)
    # Fill the probabilities of the classes that the model can predict
    probabilities[model.classes_] = predicted_probabilities
    return probabilities