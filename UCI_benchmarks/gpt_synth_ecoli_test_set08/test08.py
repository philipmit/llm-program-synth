import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Load the dataset
ecoli = pd.read_csv(
    '/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', 
    delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Prepare feature (X) and target (y) arrays
X = ecoli.iloc[:, 1:-1].values  
y = ecoli.iloc[:, -1].values 
# Encode class labels as numbers
label_encoder = LabelEncoder()
y_labeled = label_encoder.fit_transform(y)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_labeled, test_size=0.5, random_state=42)
# Standardize features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Use Logistic Regression for prediction
lr_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
lr_model.fit(X_train, y_train)
# Define the predict_label function
def predict_label(raw_data):
    # Assume the input raw_data is a list with 7 elements
    # Convert it to a numpy array with shape (1, 7)
    raw_data = np.array(raw_data).reshape(1, -1)
    # Standardize the features
    data = scaler.transform(raw_data)
    # Get probabilities as output of the Logistic Regression model
    probas = lr_model.predict_proba(data)
    return probas