import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# Load the ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Get features and labels
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the first and last 
y = ecoli.iloc[:, -1]    # All rows, only the last column
# Replace string labels with numeric labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# Convert pandas DataFrame to NumPy arrays
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Initialize Logistic Regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)
# Function to predict labels
def predict_label(sample):
    # The sample is already all features, without the labels or 'Sequence Name'
    sample_reshaped = np.array(sample).reshape(1, -1)
    # Calculate probability for each class
    predicted_probabilities = model.predict_proba(sample_reshaped)
    # Return the array of probabilities
    return predicted_probabilities[0]