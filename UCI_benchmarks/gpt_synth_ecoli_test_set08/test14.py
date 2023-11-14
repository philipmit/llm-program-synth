import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', 
                    delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# All rows, all columns except the last one for features
# All rows, only the last column for the target
X = ecoli.iloc[:, 1:-1].values  
y = ecoli.iloc[:, -1].values
# replace strings with numbers in y
unique_classes = np.unique(y)
for i, class_name in enumerate(unique_classes):
    y[y == class_name] = i
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42)
# Train a Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000, multi_class='ovr')
log_reg.fit(X_train, y_train)
def predict_label(raw_data):
    """ Predict probabilities for a given raw unprocessed data. """
    raw_data = np.array(raw_data).reshape(1, -1)
    probabilities = log_reg.predict_proba(raw_data)
    return probabilities[0]