import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Load dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Split into predictors and target
X = ecoli.iloc[:, 1:-1]
y = ecoli.iloc[:, -1]
# Convert class labels to numeric
y_labels = np.unique(y)
y = y.replace(y_labels, np.arange(len(y_labels)))
# Split the data, stratifying on y to ensure balance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
# Standardize the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
# Define and train model
model = LogisticRegression(multi_class='ovr', max_iter=200)
model.fit(X_train, y_train)
# Define prediction function
def predict_label(raw_sample):
    # Reshape the sample
    raw_sample = raw_sample.reshape(1, -1)
    # Standardize the raw sample
    raw_sample = scaler.transform(raw_sample)
    # Predict the probabilities for each class
    probabilities = model.predict_proba(raw_sample)
    return probabilities[0]