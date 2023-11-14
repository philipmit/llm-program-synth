from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1].values  # All rows, all columns excluding 'Sequence Name' and 'class'; Features
y = ecoli.iloc[:, -1].values    # All rows, only last column, 'class'; Target labels
# Replace string labels with unique integers
unique_classes, y = np.unique(y, return_inverse=True)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Scale the data; Fit the scaler only on the training set for unbiased transformation
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
# Train the logistic regression model
lr = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=4000)
lr.fit(X_train, y_train)
# Define the function to predict class probabilities for a single raw sample
def predict_label(single_sample):
    # Preprocess the sample just as we did the training data
    processed_sample = scaler.transform(np.array(single_sample).reshape(1, -1))
    # Get predicted probabilities from the model
    predicted_probabilities = lr.predict_proba(processed_sample)
    # Ensure probabilities for all classes by manually zero-filling for missing classes
    all_probs = np.zeros(len(unique_classes))
    all_probs[lr.classes_] = predicted_probabilities
    return all_probs