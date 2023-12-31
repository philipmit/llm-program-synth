import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]
y = ecoli.iloc[:, -1]
# Replace strings with numbers in y
classes = list(np.unique(y))
y = y.replace(classes, list(range(len(classes))))
# Convert to numpy arrays
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train the RandomForestClassifier model. 
# It is a versatile machine learning model and can often provide a high AUC.
# We will also increase the number of estimators and limit the max_depth of the trees to prevent overfitting.
clf = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=42)
clf.fit(X_train, y_train)
# Define a function that predicts label for new data
def predict_label(sample):
    sample = np.array(sample).reshape(1,-1)
    # Apply the same scaling to the sample
    sample = scaler.transform(sample)
    # Function returns a list of prediction probabilities
    probabilities = clf.predict_proba(sample)
    padding = [0]*(len(classes)-len(probabilities[0]))
    probabilities = np.append(probabilities,padding)
    return probabilities