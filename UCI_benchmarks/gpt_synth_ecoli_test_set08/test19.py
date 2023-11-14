import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# Replace strings with numbers in y
y = y.replace(list(np.unique(y)), [0, 1, 2, 3, 4, 5, 6, 7])
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Scaling the data for normalization to improve model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Applying Logistic Regression with multi_class parameter set to 'ovr' for binary problems 
lr_model = LogisticRegression(random_state=42, multi_class='ovr')
lr_model.fit(X_train, y_train)
# Define the predict_label function
def predict_label(sample):
    global scaler, lr_model
    sample = np.array(sample).reshape(1, -1)
    sample = scaler.transform(sample)
    prediction = lr_model.predict_proba(sample)
    # Generate a probabilities array for all possible classes
    full_probs = [0]*8
    for idx, prob in enumerate(prediction[0]):
        full_probs[idx] = prob
    return full_probs