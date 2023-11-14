import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# Load the dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', 
                    delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]    # All rows, only the last column
# replace strings with numbers in y
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# make a OneHotEncoder and fit it to the known classes
enc = OneHotEncoder()
enc.fit(y.reshape(-1, 1))
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Initialize Logistic Regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
# Fit the model to the training data
model.fit(X_train, y_train)
def predict_label(sample):
    # the predict_proba function returns probabilities per class
    proba = model.predict_proba(sample.reshape(1, -1))
    # we need to duplicate the probability array to match the same form as the OneHotEncoder
    proba = np.repeat(proba, enc.transform([[0]]).shape[1])
    return proba