from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Fit logistic regression model on training data
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)
def predict_label(raw_data):
    """
    Function to predict the label of a single sample of raw, unprocessed iris data.
    :param raw_data: A list containing the iris properties.
    :return: A list of probabilities of the sample belonging to each class.
    """
    # Ensure the input data is reshaped into a 2D array for the predict_proba function
    probabilities = log_reg_model.predict_proba(raw_data.reshape(1, -1))
    # Here is the fix: return only the list of probabilities and not the entire numpy array.
    return probabilities[0]