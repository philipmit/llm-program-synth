import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', 
                    delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Prepare the feature matrix and target vector
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# Replace strings with numbers in y 
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7])
# Convert DataFrame to numpy array
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Initialize the StandardScaler
scaler = StandardScaler()
# Fit the StandardScaler with the training data
X_train = scaler.fit_transform(X_train)
# Train the logistic regression model
log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
log_reg.fit(X_train, y_train)
def predict_label(sample):
    # Process the raw unprocessed data for a single sample
    sample = np.array(sample).reshape(1, -1)
    # Standardize the features 
    sample = scaler.transform(sample)
    # Return the predicted probabilities for that sample
    return log_reg.predict_proba(sample)[0]  # Select the first item of the multi-dimensional array 