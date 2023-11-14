import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
# Load the dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Convert categorical target to numerical values
y = ecoli.iloc[:, -1].values
unique_y = np.unique(y)
mapping = {y:i for i, y in enumerate(unique_y)}
y = np.array([mapping[i] for i in y])
# Features
X = ecoli.iloc[:, 1:-1].values
# Split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardization of the features
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
# Increase the number of trees and reduce the learning rate
parameters = {
    "loss":["deviance"],
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth":[3,5,8],
    "max_features":["log2","sqrt"],
    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":[10, 50, 100, 150]
    }
gb = GradientBoostingClassifier(random_state=42)
clf = GridSearchCV(gb, parameters, cv=5, n_jobs=-1)
clf.fit(X_train, y_train)
# Extract best_model
best_model = clf.best_estimator_
def predict_label(input_data):
    # input_data should be raw data for single sample, array of the format [mcg, gvh, lip, chg, aac, alm1, alm2]
    # Scale the input
    input_data = scaler.transform([input_data])
    # Return the predicted probabilities for each class, ensuring there is a probability for each class in y_train
    prediction = best_model.predict_proba(input_data)[0]
    full_prediction = np.zeros(len(unique_y))
    for index, prob in zip(best_model.classes_, prediction):
        full_prediction[index] = prob
    return full_prediction