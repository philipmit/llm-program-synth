import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, split columns excluding the first and the last one
y = ecoli.iloc[:, -1]    # All rows, only the last column (class label)
# replace string labels with numbers in y
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7])
X = X.to_numpy()
y = y.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)
def predict_label(input_data):
    """
    Function to predict the label for given input data using trained logistic regression model.
    input_data should be a pandas DataFrame.
    """
    input_data_features = input_data.iloc[1:-1]  # Extract features in the same way as we did with the entire ecoli dataset
    input_data_features = np.array(input_data_features).reshape(1, -1)    # reshape the data for prediction
    return lr.predict_proba(input_data_features)[0]