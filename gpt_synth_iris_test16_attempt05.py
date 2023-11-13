from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Create a new logistic regression model
log_reg = LogisticRegression(max_iter=200)
# Train the model using the training dataset
log_reg.fit(X_train, y_train)
def predict_label(raw_data):
    """
    Function to predict the label of a single sample from the raw unprocessed data
    """
    # The model predicts the probabilities of the input sample being in each class
    probabilities = log_reg.predict_proba([raw_data])
    return probabilities[0]