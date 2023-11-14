import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Prepare the data 
X = ecoli.iloc[:, 1:-1].values  
y = ecoli.iloc[:, -1].values 
# Transform y from categorical to numerical values
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5, random_state=42)
# Train a Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000, multi_class='ovr')
log_reg.fit(X_train, y_train)
def predict_label(raw_data):
    """ 
    Predict probabilities for a given raw unprocessed data. 
    """
    raw_data = np.array(raw_data).reshape(1, -1)
    probabilities = log_reg.predict_proba(raw_data)
    return probabilities[0]