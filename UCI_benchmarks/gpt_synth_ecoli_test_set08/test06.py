import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
# Load the ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Pre-process the dataset
X = ecoli.iloc[:, 1:-1].values  # All rows, all the columns except the last one
y = ecoli.iloc[:, -1].values # All rows, only the last column
# Replace string labels with numbers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Define logistic regression model and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# Explicitly providing unique classes to model
model.classes_ = np.unique(y)
# Define function to predict labels
def predict_label(X_sample):
    # reshaping the input to match the input shape for sklearn ML models
    X_sample = np.array(X_sample).reshape(1, -1)
    # getting the model's predicted probabilities with predict_proba
    pred_proba = np.zeros((1,len(model.classes_)))
    pred_proba[0][model.predict(X_sample)[0]] = 1
    return pred_proba[0]