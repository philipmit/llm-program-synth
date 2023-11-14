import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# Replace string-type class labels with integer labels
classes = list(np.unique(y))
y = y.replace(classes, range(len(classes)))
# Convert the pandas DataFrame to numpy array
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize the training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Attempt to improve validation AUC by using GradientBoostingClassifier which generally provides better performance than RandomForestClassifier
# Adjust the learning rate, n_estimators, and max_depth parameters to achieve optimal results
model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=500, max_depth=5, random_state=42)
model.fit(X_train, y_train)
def predict_label(sample):
    # Reshape sample data into correct dimensions and scale it
    sample = np.array(sample).reshape(1, -1)
    sample = scaler.transform(sample)
    # Get class probabilities
    probabilities = model.predict_proba(sample)[0]
    # Manually append zeros to probabilities array if necessary
    if len(probabilities) < len(classes):
        probabilities = np.append(probabilities, [0]*(len(classes) - len(probabilities)))
    return probabilities