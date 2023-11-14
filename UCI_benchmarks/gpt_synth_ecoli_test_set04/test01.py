import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Create feature matrix and target array
X = ecoli.iloc[:, 1:-1]  
y = ecoli.iloc[:, -1]
# List to keep track of unique classes
unique_classes = list(np.unique(y))
# Convert string labels in y to numeric
y = y.replace(unique_classes, list(range(len(unique_classes))))
# Convert pandas objects to numpy arrays
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Train the logistic regression model with new parameters
lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.9, penalty='l1', solver='liblinear', max_iter=5000))
lr.fit(X_train, y_train)
def predict_label(input_data):
    """
    Function to predict the label for given input data using trained logistic regression model.
    input_data should be a numpy array.
    """
    # Initialize a prediction list of 0's with a length equal to the total number of classes
    prediction_probs = [0 for _ in range(len(unique_classes))]
    # Reshape the input data for prediction
    input_data_features = input_data.reshape(1, -1)
    # Perform prediction
    predicted_probabilities = lr.predict_proba(input_data_features)[0]
    # Get classes that the model was trained on
    trained_classes = lr.classes_
    for i in range(len(trained_classes)):
        prediction_probs[trained_classes[i]] = predicted_probabilities[i]
    return prediction_probs