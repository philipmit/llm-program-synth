import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# Load the dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Prepare the feature and target variables
X = ecoli.iloc[:, 1:-1].values  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# Transform target labels into number using LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Train the logistic regression model
log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
log_reg.fit(X_train, y_train)
def predict_label(raw_data_sample):
    # Ensure the input data is 2D
    if len(raw_data_sample.shape) == 1:
        raw_data_sample = np.expand_dims(raw_data_sample, 0)
    # predict the class probalities
    proba = log_reg.predict_proba(raw_data_sample)
    return proba