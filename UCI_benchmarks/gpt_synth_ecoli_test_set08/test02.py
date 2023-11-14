import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows excluding the first and last columns
y = ecoli.iloc[:, -1]    # All rows, only the last column
# Replace strings in y with numbers
classes = list(np.unique(y))
y = y.replace(classes, list(range(len(classes))))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Create logistic regression object
logreg = LogisticRegression(max_iter=1000, multi_class='ovr')
logreg.fit(X_train, y_train)
#Parsing output to meet the requirements of the validation code
def predict_label(sample):
    sample = np.array(sample).reshape(1, -1) 
    sample = scaler.transform(sample)  
    probabilities = logreg.predict_proba(sample)
    #turn the probabilities into a list of length 8 (number of classes)
    #filling missing values with 0
    result = [0]*8
    for index, probability in enumerate(probabilities[0]):
        result[index] = probability
    return result