from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Train a Logistic Regression model on the training set
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
def predict_label(new_data):
    """
    Function to predict the probabilities of the Iris labels for a new sample.
    Args:
    new_data: A 1-D array of feature values for a new sample.
    Returns:
    The predicted probabilities of the labels for the new sample.
    """
    return log_reg.predict_proba([new_data])