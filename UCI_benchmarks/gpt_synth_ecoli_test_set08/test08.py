import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
# Load the dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Prepare feature (X) and target (y) arrays
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
# Use XGBoost for prediction, with a high number of rounds and a learning rate adapted for better performance
params = {
    "objective":"multi:softprob",
    "max_depth":8,
    "silent":1,
    "eta":0.01, # learning rate
    "num_class":8,
    "n_estimators":600
}
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test)
xgb_model = xgb.train(params, dtrain, num_boost_round=800)
# Define the predict_label function
def predict_label(X):
    # Assume the input X is a list with 7 elements
    # Convert it to a numpy array with shape (1, 7)
    X = np.array(X).reshape(1, -1)
    # Standardize the features
    X = scaler.transform(X)
    # Transform it to DMatrix for XGBoost
    ddata = xgb.DMatrix(data = X)
    # Get probabilities as output of the XGBoost model
    probas = xgb_model.predict(ddata)
    return probas