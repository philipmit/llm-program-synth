import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Extract features and target variable
X = ecoli.iloc[:, 1:-1].values
y = ecoli.iloc[:, -1].replace(np.unique(y), range(len(np.unique(y)))).values
# Split the loaded dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize the training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Initialize Logistic Regression model with the 'lbfgs' solver, and fit it to the training data
model = LogisticRegression(solver='lbfgs', multi_class='multinomial', n_jobs=-1, class_weight='balanced')
model.fit(X_train, y_train)
def predict_label(sample):
    # Convert the sample data to array, reshape and standardize it
    sample = np.array(sample).reshape(1, -1)
    sample = scaler.transform(sample)
    # Return the predicted probabilities for that sample
    return model.predict_proba(sample)[0]