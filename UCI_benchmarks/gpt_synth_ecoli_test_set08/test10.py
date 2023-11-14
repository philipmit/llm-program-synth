import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# Load the dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:]  # All rows, all columns except the first 
y = ecoli.iloc[:, -1]    # All rows, only the last column
# replace strings with numbers in y
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7])
# Convert pandas DataFrame to NumPy arrays
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Initialize Logistic Regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
# Fit the model to the training data
model.fit(X_train, y_train)
# Function to predict labels
def predict_label(single_sample):
    # Take only the features (all columns except the first) in single_sample, as the model was trained on those
    sample_features = single_sample[1:]
    # Reshape the sample
    sample_reshaped = np.array(sample_features).reshape(1, -1)
    # Calculate probability for each class
    predicted_probabilities = model.predict_proba(sample_reshaped)
    # Return the array of probabilities
    return predicted_probabilities[0]