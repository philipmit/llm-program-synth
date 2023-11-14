import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# replace strings with numbers in y
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7])
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
# classifier with grid search
param_grid = { 
    'n_estimators': [50, 100, 200],
    'max_depth' : [4,6,8],
    'criterion' :['gini', 'entropy']
}
rfc=RandomForestClassifier(random_state=42)
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5, scoring='roc_auc_ovr')
CV_rfc.fit(X_train, y_train)
# print out optimal parameters
print(CV_rfc.best_params_)
# Build model pipeline with best parameters
rf_tuned = make_pipeline(StandardScaler(),
                         RandomForestClassifier(n_estimators=CV_rfc.best_params_['n_estimators'], 
                                                max_depth=CV_rfc.best_params_['max_depth'],
                                                criterion=CV_rfc.best_params_['criterion'],
                                                random_state=42))
rf_tuned.fit(X_train, y_train)
def predict_label(sample):
    """
    Predict the class probabilities for a given sample.
    Parameters:
    sample -- the raw unprocessed data for a sample
    Returns: 
    the predicted probabilities for the sample
    """
    return rf_tuned.predict_proba([sample])[0]