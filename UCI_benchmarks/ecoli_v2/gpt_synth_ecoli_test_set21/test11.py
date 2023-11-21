#<PrevData>
######## Load and preview the dataset and datatypes
# Import necessary libraries
import pandas as pd

# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)

# Rename columns
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']

# Print the shape of the dataset and the first few rows
print(ecoli.shape)
print(ecoli.head())
print(ecoli.dtypes)
print(ecoli.isnull().sum())
#</PrevData>

#<PrepData>
######## Prepare the dataset for training
# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Define features and target
X = ecoli.iloc[:, 1:-1].values  # exclude the Sequence Name and target column
y = ecoli.iloc[:, -1].values  # the target column

# Perform label encoding for the target to convert string class labels to integers
le = LabelEncoder()
y = le.fit_transform(y)

# Split the dataset into training (50%) and test sets (50%) with random_state=42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# We'll keep a copy of the unscaled X_test to use when making predictions
unscaled_X_test = X_test.copy()

# Scale the training set
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

# Scale the test set using the same scaler
X_test = sc.transform(X_test)
#</PrepData>

#<Train>
######## Train the model using the training data
# Import the LogisticRegression function from sklearn
from sklearn.linear_model import LogisticRegression

# Initialize the logistic regression model
lr = LogisticRegression(max_iter=500, multi_class='ovr', solver='liblinear', random_state=42)

# Fit the model on the training data
lr.fit(X_train, y_train)
#</Train>

#<Predict>
######## The predict_label function will use the trained logistic regression model to predict the label of a new sample
def predict_label(sample):
    # Sample is expected to be a list of 7 decimal values
    # Reshape the sample to (1,-1) and standardize it using the same scaler as for the training set
    sample = sc.transform(sample.reshape(1,-1))
    
    # Use the trained model to predict the probabilities
    probabilities = lr.predict_proba(sample)
    
    # Since the output is a 2D array with shape (1,8) we need to take the first element to get the (8,) shape
    probabilities = probabilities[0]
    
    # Return the probabilities as a list
    return probabilities.tolist()
#</Predict>
