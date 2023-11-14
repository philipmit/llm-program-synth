import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
# Load the ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Define Predictor variables and target variable
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# Apply LabelEncoder() to the target variable
le = LabelEncoder()
y = le.fit_transform(y)
# Split the dataset into training and testing sets using stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
# Standardize the features
scaler = StandardScaler()
# Fit on the training data and transform both the training and the test data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Fit the logistic regression model
clf = LogisticRegression(multi_class='ovr', class_weight='balanced', solver='liblinear')
clf.fit(X_train, y_train)
def predict_label(raw_data):
    """
    This function takes raw unprocessed data for a single sample and returns predicted probabilities for that sample.
    """
    raw_data_sc = scaler.transform(np.array(raw_data).reshape(1, -1))
    # predict probabilities
    predicted_probabilities = clf.predict_proba(raw_data_sc)
    return predicted_probabilities[0]