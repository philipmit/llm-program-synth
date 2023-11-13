import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]    # All rows, only the last column
# Replace strings with numbers in y
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7])
X=X.to_numpy()
y=y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize the Features Matrix
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Define the logistic regression model
model = LogisticRegression(multi_class='ovr', max_iter=1000)
# Fit the model on the training data
model.fit(X_train, y_train)
def predict_label(raw_sample):
    # Verify if the sample is in the correct shape, if not reshape it
    if len(raw_sample.shape) == 1:
        raw_sample = raw_sample.reshape(1, -1)
    # Preprocess the raw input
    processed_sample = scaler.transform(raw_sample)
    # Use the trained model to predict the label
    predicted_probs = model.predict_proba(processed_sample)
    return predicted_probs