#<PrevData>
print('********** Load and preview the dataset and datatypes')
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Read file
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)

# Define features, X, and labels, y
X = df.iloc[:, 1:-1]  # All rows, all columns except the first and last one
y = df.iloc[:, -1]  # All rows, only the last column

# Convert labels to unique integers for model compatibility
y = y.replace(list(np.unique(y)), list(range(len(np.unique(y)))))

# Check if any class contain less than 2 instances
min_class_size = y.value_counts().min()
if min_class_size < 2:
    print(f"The smallest class contains {min_class_size} instance(s). Stratified split is not possible.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
else:
    # Convert dataframe to numpy for model compatibility
    X = X.apply(pd.to_numeric).to_numpy()  # Convert string to numeric
    y = y.to_numpy()

    # Split the dataset into training and testing sets. Set stratify=y for equal distribution in train/test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)

# Scale the features to standardize them. Fit only to the training data, then apply the transformations to the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#</PrevData>

#<PrepData>

# Grid search to optimize hyperparameters
parameters = {'penalty': ['l1', 'l2', 'elasticnet', 'none'], 
              'C': np.logspace(-4, 4, 20), 
              'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
              'max_iter': list(range(100,800,100))}
  
log_reg = LogisticRegression(random_state=42)
clf = GridSearchCV(log_reg, parameters, cv=5, verbose=0, n_jobs=-1)
best_model = clf.fit(X_train,y_train)
params_optimal = best_model.best_params_

print('Best penalty:', params_optimal['penalty'])
print('Best C:', params_optimal['C'])
print('Best solver:', params_optimal['solver'])
print('Best max_iter:', params_optimal['max_iter'])

#</PrepData>

#<Train>
# Initialize Logistic Regression model with optimal parameters.
model_optimal = LogisticRegression(penalty = params_optimal['penalty'], 
                                   C = params_optimal['C'], 
                                   solver = params_optimal['solver'], 
                                   max_iter = params_optimal['max_iter'],
                                   random_state=42)

# Fit the model using training data.
model_optimal.fit(X_train, y_train)

# Use trained model to predict labels of the test set
y_pred_proba = model_optimal.predict_proba(X_test)[::,1]

# Calculate AUC
auc = roc_auc_score(y_test, y_pred_proba)
print('Improved AUC:', auc)
#</Train>

#<Predict>
print('Define the prediction function with optimal params')
# Define the prediction function
def predict_label_opt(one_sample):
    # Confirm the input data is in the right format (list). If not, convert it to list
    if isinstance(one_sample, np.ndarray):
        one_sample = one_sample.tolist()

    if not isinstance(one_sample[0], list):
        one_sample = [one_sample]

    # Convert list to numpy array
    one_sample = np.array(one_sample)
    # Scale the features of the one_sample to standardize them
    one_sample = sc.transform(one_sample)
    # Use predict_proba instead of predict to get probabilities 
    return model_optimal.predict_proba(one_sample)[0]  # Return the first element to keep it within 2 dimensions
#</Predict>
