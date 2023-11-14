import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
# Load the ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# All rows, all columns except the last one
X = ecoli.iloc[:, 1:-1].to_numpy()
# All rows, only the last column
y = ecoli.iloc[:, -1]
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7]).to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Train the Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train) 
def predict_label(data):
    # Assumes data is a np array with shape (no. of features)
    preprocessed_data = data.reshape(1, -1)
    # Predict probabilities and squeeze off the extra dimension
    probabilities = np.squeeze(log_reg.predict_proba(preprocessed_data))
    return probabilities