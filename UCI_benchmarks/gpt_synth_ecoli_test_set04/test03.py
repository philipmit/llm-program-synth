import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Load the dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Preprocess
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1] # All rows, only the last column
labels = list(np.unique(y))
y = y.replace(labels, list(range(len(labels))))
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Scale
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
# One versus Rest Logistic Regression
multi_class_regressor = LogisticRegression(max_iter=1000, multi_class='ovr').fit(X_train, y_train)
def predict_label(sample): 
    # Make sure to apply the same scaling to the sample
    sample = scaler.transform([sample])
    # Use the logistic regression model to predict the label for the sample
    predicted_label_probabilities = multi_class_regressor.predict_proba(sample)
    return predicted_label_probabilities[0]