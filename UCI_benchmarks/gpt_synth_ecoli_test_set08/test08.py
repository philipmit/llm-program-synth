import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
# Load dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1].values  # All rows, all columns except last one
y = ecoli.iloc[:, -1].values   # All rows, only the last column
# Use LabelEncoder to encode class labels as numbers
le = LabelEncoder()
all_classes = np.unique(y)
num_classes = len(all_classes)
y_encoded = le.fit_transform(y)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.5, random_state=42)
# Train logistic regression
logreg = LogisticRegression(random_state=42, multi_class='multinomial', solver='lbfgs', max_iter=10000)
logreg.fit(X_train, y_train)
# Define the predict_label function
def predict_label(X):
    # Assume the input x is a list with 7 elements
    # Convert it to (1, 7) shape numpy.ndarray
    X = np.array(X).reshape(1, -1)
    # Get probabilities as output of the logistic regression model
    probas = logreg.predict_proba(X)[0]
    # Create a zero-filled array to hold probabilities for all classes
    all_probas = np.zeros(num_classes)
    # Fill in the probabilities for the classes that the model was trained on
    all_probas[logreg.classes_] = probas
    return all_probas