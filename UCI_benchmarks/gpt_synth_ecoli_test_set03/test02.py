import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Random Seed for reproducibility
np.random.seed(42)
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Prepare feature matrix X and target vector y
X = ecoli.iloc[:, 1:-1]
y = ecoli.iloc[:, -1].replace(np.unique(y), list(range(len(np.unique(y)))))
# Split the dataset into training and testing sets using stratified split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Normalization of values in train set for improved model convergence
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Creating an instance of Logistic Regression with increased iteration count and L2 regularization
model = LogisticRegression(max_iter=1000, penalty='l2')
# Fitting the data to the model
model.fit(X_train, y_train)
# Define the function
def predict_label(raw_data):
    # Data preprocessing
    ready_data = scaler.transform(raw_data.reshape(1, -1))
    # Perform prediction
    probabilities = model.predict_proba(ready_data)
    return probabilities