import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# Load dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# preparing feature (X) and target (y) arrays
X = ecoli.iloc[:, 1:-1].values  
y = ecoli.iloc[:, -1].values 
# Encode class labels as numbers
label_encoder = LabelEncoder()
y_labeled = label_encoder.fit_transform(y)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_labeled, test_size=0.2, random_state=42)
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Use RandomForestClassifier to predict
random_forest = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
random_forest.fit(X_train, y_train)
# Define the predict_label function
def predict_label(X):
    # Assume the input X is a list with 7 elements
    # Convert it to (1, 7) shape numpy.ndarray
    X = np.array(X).reshape(1, -1)
    # Standardize the features
    X = scaler.transform(X)
    # Get probabilities as output of the Random Forest model
    probas = random_forest.predict_proba(X)[0]
    return probas