#<PrevData>
######## Load and preview the dataset and datatypes
# Import necessary libraries
import pandas as pd

# Read file
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

# We have to remember to transform the test set as well to make sure our prediction function works correctly
X_test = sc.transform(X_test)

# Scale the inputs in the training set
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
#</PrepData>

#<Train>
######## Train the model using the training data
from sklearn.linear_model import LogisticRegression

# Create logistic regression
# As the target has multiple labels, we have to set multi_class parameter to 'ovr' which stands for 'one vs rest'
log_reg = LogisticRegression(multi_class='ovr', max_iter=500, random_state=42)

# Train the model using the training data
log_reg.fit(X_train, y_train)
#</Train>

#<Predict>
######## The predict_label function will use the trained logistic regression model to predict the label of a new sample
def predict_label(sample):
    # Convert the sample to a numpy array and reshape it to (1,-1)
    sample = np.array(sample).reshape(1,-1)
    
    # Standardize the sample
    sample = sc.transform(sample)
    
    # Use the trained model to predict the probabilities
    pred_prob = log_reg.predict_proba(sample)
    
    # Convert numpy array to list and return
    return pred_prob[0].tolist()
#</Predict>
