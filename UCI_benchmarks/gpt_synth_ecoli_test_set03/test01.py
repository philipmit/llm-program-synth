import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
# Load the dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Define features and target
X = ecoli.iloc[:, 1:-1]
y = ecoli.iloc[:, -1]
# Replace strings with numbers in y
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# Convert to numpy arrays
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Train the Logistic Regression model
lr = LogisticRegression(max_iter=1000, multi_class='ovr')
lr.fit(X_train, y_train)
def predict_label(data):
    data = np.array(data).reshape(1, -1)
    proba = lr.predict_proba(data)[0]
    # Ensures predicted probabilities are for all classes
    full_proba = np.zeros(len(np.unique(y)))
    # Index of predicted classes are found after ranking 
    classes_ = lr.classes_
    ranks = label_encoder.transform(classes_)
    # Setting prediction only for predicted classes
    for rank, prob in zip(ranks, proba):
        full_proba[rank] = prob
    return full_proba