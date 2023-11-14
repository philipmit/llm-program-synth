import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Load the dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# Replace strings with numbers in y
unique_labels = np.unique(y)
y = y.replace(unique_labels, range(len(unique_labels)))
X=X.to_numpy()
y=y.to_numpy()
 #Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Create logistic regression object
logreg = LogisticRegression(multi_class='ovr', max_iter=10000)
# Train the model using the training sets
logreg.fit(X_train, y_train)
def predict_label(raw_data):
    # Predict probabilities
    probabilities = logreg.predict_proba(raw_data.reshape(1, -1))
    return probabilities[0]