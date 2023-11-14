import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
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
np.unique(y)
len(list(np.unique(y)))
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7])
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets using Stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
# Quick grid search to find optimal C, solver and multi_class parameters
param_grid = [{'C': [0.01, 0.1, 1, 10, 100], 
               'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
               'multi_class' : ['auto', 'ovr', 'multinomial']}]
log_reg = LogisticRegression(max_iter=3000, penalty='l2', random_state=42)
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
# Print best parameters after tuning
print(grid_search.best_params_)
# Build model pipeline
lr_tuned = make_pipeline(StandardScaler(),
                          LogisticRegression(solver=grid_search.best_params_['solver'], 
                                             multi_class=grid_search.best_params_['multi_class'], 
                                             max_iter=3000, 
                                             penalty='l2', 
                                             C=grid_search.best_params_['C'], 
                                             random_state=42))
lr_tuned.fit(X_train, y_train)
def predict_label(sample):
    """
    Predict the class probabilities for a given sample.
    Parameters:
    sample -- the raw unprocessed data for a sample
    Returns: 
    the predicted probabilities for the sample
    """
    return lr_tuned.predict_proba([sample])[0]