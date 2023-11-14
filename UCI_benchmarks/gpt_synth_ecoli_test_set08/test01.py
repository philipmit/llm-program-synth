import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]
y = ecoli.iloc[:, -1]
# Replace strings with numbers in y
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7])
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train the logistic regression model
clf = LogisticRegression(random_state=0, multi_class='ovr')
clf.fit(X_train, y_train)
# Define a function that predicts label for new data
def predict_label(sample):
    sample = np.array(sample).reshape(1,-1)
    sample = scaler.transform(sample)  # We need to apply the same scaling to the new data
    probabilities = clf.predict_proba(sample)
    print(probabilities)
    # As we need to provide a list of probabilities, if it's a single class, we reshape to have a 2D list
    probabilities = probabilities.reshape(-1, len(probabilities[0])).tolist()
    # Flatten the list of probabilities as roc_auc_score expects a 2D input but one-dimensional output.
    probabilities = [prob for sublist in probabilities for prob in sublist]
    return probabilities