import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
# Hyperparameters tuning using GridSearchCV
parameters = {'max_depth': [5, 10, 15], 'n_estimators': [50, 100, 200]}
rf = RandomForestClassifier(random_state=42)
clf = GridSearchCV(rf, parameters)
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