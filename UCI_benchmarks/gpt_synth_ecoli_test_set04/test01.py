import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
# Load the Ecoli data
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Create feature matrix (X) and target array (Y)
X = ecoli.iloc[:, 1:-1]
y = ecoli.iloc[:, -1]
# List to keep track of unique classes
unique_classes = list(np.unique(y))
# Convert string labels in y to numeric
y = y.replace(unique_classes, list(range(len(unique_classes))))
# Convert pandas objects to numpy arrays
X, y = X.to_numpy(), y.to_numpy()
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Defining logistic regression model with a pipeline for scaling
model = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(solver='liblinear'))
])
# Setting up a grid of parameters to optimize
params = {
    "lr__C":np.logspace(-3,3,7),
    "lr__penalty":["l1", "l2"]
}
# Performing Grid Search to find the best parameters
grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5)
grid_search.fit(X_train, y_train)
def predict_label(input_data):
    """
    Function to predict the label for given input data using the optimized logistic regression model.
    input_data should be a numpy array.
    """
    prediction_probs = [0 for _ in range(len(unique_classes))]
    input_data_features = input_data.reshape(1, -1)
    predicted_probabilities = grid_search.predict_proba(input_data_features)[0]
    trained_classes = grid_search.best_estimator_.named_steps['lr'].classes_
    for i in range(len(trained_classes)):
        prediction_probs[trained_classes[i]] = predicted_probabilities[i]
    return prediction_probs