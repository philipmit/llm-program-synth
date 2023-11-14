mport pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
# Load the dataset
ecoli = pd.read_csv('ecoli.data', header=None, sep="\s+")
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Convert categorical target to numerical values
y = ecoli.iloc[:, -1]
unique_values = np.unique(y)
y = y.replace(unique_values, np.arange(len(unique_values)))
# Features
X = ecoli.iloc[:, 1:-1]
# Convert to numpy arrays
X = X.values
y = y.values
# Split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Scaling the features as logistic regression requires features to be on similar scale
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
# Creating and Training the Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=5000).fit(X_train, y_train)
# prediction function
def predict_label(input_data):
    # input_data should be a list of the format [mcg, gvh, lip, chg, aac, alm1, alm2]
    # Scale the input
    input_data = scaler.transform([input_data])
    return model.predict_proba(input_data)