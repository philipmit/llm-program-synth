import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Create feature matrix and target array
X = ecoli.iloc[:, 1:-1]  
y = ecoli.iloc[:, -1]
# Convert string labels in y to numeric
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7])
# Convert pandas objects to numpy arrays
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Train the logistic regression model
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)
def predict_label(input_data):
    """
    Function to predict the label for given input data using trained logistic regression model.
    input_data should be a numpy array.
    """
    input_data_features = input_data[1:-1]  # Extract features in the same way as we did with the entire ecoli dataset
    input_data_features = np.array(input_data_features).reshape(1, -1)    # reshape the data for prediction
    return lr.predict_proba(input_data_features)[0]