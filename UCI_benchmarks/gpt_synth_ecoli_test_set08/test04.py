import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', 
                    delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# replace strings with numbers in y
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7])
# Convert dataframe to numpy array
X= X.to_numpy()
y= y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Scale the datasets using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Changed model to Gradient Boosting Classifier to possibly get a more complex decision boundary and improve accuracy.
# Tuned parameters such as learning_rate, n_estimators and max_depth, which help in reducing overfitting of model and improving performance.
model = GradientBoostingClassifier(learning_rate=0.01, n_estimators=500, max_depth=5, random_state=42)
model.fit(X_train, y_train)
# Function to predict label
def predict_label(X_raw):
    num_class = len(np.unique(y))  # Compute the number of unique classes in the dataset
    X_raw_np_array = np.array(X_raw, dtype=float).reshape(1, -1)
    X_raw_scaled = scaler.transform(X_raw_np_array)  
    # Compute prediction probabilities
    prediction = model.predict_proba(X_raw_scaled)[0]
    # Handle case where prediction probabilities have less classes than provided in dataset
    if len(prediction) != num_class:
        diff = num_class - len(prediction)  # Compute the difference between the number of classes in dataset and classes predicted
        return np.concatenate([prediction, np.zeros(diff)])  # Append zeros at the end of the prediction array to match class count 
    else:
        return prediction