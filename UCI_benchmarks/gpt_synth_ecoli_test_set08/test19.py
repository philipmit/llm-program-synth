import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
# Load the ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, all the columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# Replace strings with numbers in y
y = y.replace(list(np.unique(y)), [0, 1, 2, 3, 4, 5, 6, 7])
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
# Scaling the data for normalization to improve model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Apply RandomForest Classifier instead of Logistic Regression as it's known for achieving better AUC
rf_model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
rf_model.fit(X_train, y_train)
# Define the predict_label function
def predict_label(sample):
    global scaler, rf_model
    sample = np.array(sample).reshape(1, -1)
    sample = scaler.transform(sample)
    prediction = rf_model.predict_proba(sample)
    # Generate a probabilities array for all possible classes
    full_probs = [0]*8
    for idx, prob in enumerate(prediction[0]):
        full_probs[idx] = prob
    return full_probs