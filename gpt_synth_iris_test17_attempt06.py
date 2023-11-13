from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Feature scaling
sc = StandardScaler()
X = sc.fit_transform(X)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Create the logistic regression model
log_reg = LogisticRegression()
# Train the model with the training data
log_reg.fit(X_train, y_train)
def predict_label(raw_data):
    """Makes prediction using logistic regression model.
    Assumes raw_data is an array-like collection containing the features of one sample.
    Returns the probability of the predicted class.
    """
    # Rescaling raw_data with the predefined StandardScaler
    scaled_data = sc.transform([raw_data])
    # Return the predicted probabilities
    probabilities = log_reg.predict_proba(scaled_data)
    # Return the probabilities of the
    # class with the highest predicted probability
    return probabilities.max(axis=1)