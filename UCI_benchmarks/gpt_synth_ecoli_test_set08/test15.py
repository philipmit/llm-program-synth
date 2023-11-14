import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
# Train the logistic regression model
lr = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=400)
lr.fit(X_train, y_train)
def predict_label(sample):
    """
    Predict the class probabilities for a given sample.
    Parameters:
    sample -- the raw unprocessed data for a sample
    Returns: 
    the predicted probabilities for the sample
    """
    return lr.predict_proba([sample])[0]