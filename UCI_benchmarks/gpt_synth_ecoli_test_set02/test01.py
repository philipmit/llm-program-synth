import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import StandardScaler
# Load the ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Define predictor and target variables
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# replace class labels with numbers
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7])
# convert to numpy arrays
X=X.to_numpy()
y=y.to_numpy()
# scale the dataset 
sc = StandardScaler()
X = sc.fit_transform(X)
# split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
def predict_label(raw_data):
    # preprocess the raw data
    raw_data = np.array(raw_data).reshape(1, -1)
    # since the data was trained on scaled data, we need to scale the new data
    raw_data = sc.transform(raw_data)
    # get the predictions
    predicted_probabilities = model.predict_proba(raw_data)
    # return only the first row from the predicted probabilities, making it a 1D array
    return predicted_probabilities[0]