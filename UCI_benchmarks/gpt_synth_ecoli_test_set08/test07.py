import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Load the ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  
y = ecoli.iloc[:, -1]   
# Replace strings with numbers in y
y = y.replace(np.unique(y), np.arange(len(np.unique(y))))
# convert to numpy arrays
X = X.to_numpy()
y = y.to_numpy()
# Preprocess the inputs
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Train the logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X, y)
# Define prediction function
def predict_label(sample):
    # Don't forget to preprocess the new sample
    sample = scaler.transform(sample.reshape(1, -1))
    predicted_probs = model.predict_proba(sample)
    return predicted_probs[0]