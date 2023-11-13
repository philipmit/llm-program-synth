import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
# Load the ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Define predictor and target variables
X = ecoli.iloc[:, 1:-1]  
y = ecoli.iloc[:, -1]   
class_mapping = {label: index for index, label in enumerate(np.unique(y))}
y = y.replace(class_mapping)
# Convert to numpy arrays
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Create and train the MLP model with increased number of iterations and layers
model = MLPClassifier(hidden_layer_sizes=(100, 200, 100), max_iter=500,learning_rate_init=0.01)
model.fit(X_train, y_train)
def predict_label(raw_data):
    # Reshape the input data
    raw_data = np.array(raw_data).reshape(1, -1)
    # Scale the raw data
    raw_data = scaler.transform(raw_data)
    # Predict
    return model.predict_proba(raw_data)[0]
# The model performance on validation set
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:,1]
# Use accuracy_score and roc_auc_score for checking the model's performance
accur_score = accuracy_score(y_test, y_pred)
roc_score = roc_auc_score(y_test, y_pred_prob)
print('Accuracy Score:', accur_score)
print('ROC AUC Score:', roc_score)