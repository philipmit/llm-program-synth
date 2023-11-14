import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, the features
y = ecoli.iloc[:, -1]   # All rows, the class label
# replace string class labels with unique integers
class_labels = np.unique(y)
label_dict = {label: index for index, label in enumerate(class_labels)}
y = y.replace(label_dict)
# Convert the features and labels data to numpy
X = X.to_numpy()
y = y.to_numpy()
# Split the Ecoli dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Define the scaler
scaler = StandardScaler()
# Scaling the X_train data
X_train = scaler.fit_transform(X_train)
# Define XGBClassifier with initial parameters
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
# Setup Grid Search parameters
param_grid = {
    'max_depth': [3, 6, 10],
    'subsample': [0.5, 0.7, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0],
}
# Apply Grid Search on the model
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='roc_auc', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
# Update the model with the best parameters
xgb_model = grid_search.best_estimator_
def predict_label(raw_sample):
    # reshape raw_sample if it's only 1D (it's a single sample)
    if len(raw_sample.shape) == 1:
        raw_sample = raw_sample.reshape(1, -1)
    # normalize the sample
    sample = scaler.transform(raw_sample)
    # use the fitted model to predict the probabilities
    probas = xgb_model.predict_proba(sample)
    return probas