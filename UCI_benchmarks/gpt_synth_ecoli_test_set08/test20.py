import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
# Load the ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', 
                    delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1].values    # All rows, all columns except the last one
y = ecoli.iloc[:, -1].values   # All rows, only the last column
# Replace strings with numbers in y using label encoder
le = LabelEncoder()
y = le.fit_transform(y)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Create a logistic regression model and fit with training data
log_reg = LogisticRegression(max_iter=2000, multi_class='ovr')
log_reg.fit(X_train, y_train)
# Function to predict label
def predict_label(sample):
    sample = np.array(sample).reshape(1, -1)
    predicted_probabilities = log_reg.predict_proba(sample)
    # Create an array to hold probabilities for each class
    probabilities = np.zeros(len(le.classes_))
    # Assign predicted probabilities to respective index
    probabilities[log_reg.classes_] = predicted_probabilities[0]
    return probabilities