import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Define features and target
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7])
# Convert to numpy arrays
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Pre-processing: Standard Scaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Train Logistic Regression model
lr = LogisticRegression(random_state=42, max_iter=10000, multi_class='multinomial', solver='lbfgs')
lr.fit(X_train, y_train)
def predict_label(sample):
    sample = sc.transform(sample.reshape(1, -1))  # Apply the same scaling to the sample
    probabilities = lr.predict_proba(sample)[0]  # Predict probabilities for the sample
    return probabilities