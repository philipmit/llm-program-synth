#<PrevData>
print('********** Load and preview the dataset and datatypes')
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
df = pd.read_csv('ecoli.data', delim_whitespace=True, header=None)
df.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']

# Define features, X, and labels, y
X = df.iloc[:, 1:-1]  # All rows, all columns except the last one
y = df.iloc[:, -1]   # All rows, only the last column

# Preview dataset and datatypes
print(df.head())
print(df.dtypes)
print(df.info())
#</PrevData>
#<PrepData>
print('********** Prepare the dataset for training')

# Transform y from strings to integers
le = LabelEncoder()
y = le.fit_transform(y)

# Convert DataFrames to NumPy arrays
X = X.values
y = np.array(y) 

# Split the dataset into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Scale the features 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)   # use the same scaler (from training data) to standardize test data

# Train the logistic regression model
# Use GridSearchCV to optimize hyperparameters C and max_iterations.
param_grid = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"], 'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)

# Fetch best estimator
model = grid.best_estimator_
model.fit(X_train, y_train)
#</PrepData>
#<Predict>
print('********** Define a function to predict labels')
def predict_label(one_sample):
    # Standardize the one_sample to match the data model was trained on
    one_sample = scaler.transform([one_sample])
    # Use predict_proba to get class probabilities, since roc_auc_score needs probabilities
    probability_scores = model.predict_proba(one_sample)
    # Return the class probabilities not the class label 
    return list(probability_scores[0]) # The prediction should be returned as a list of probabilities, making it iterable for the validation code
#</Predict>
