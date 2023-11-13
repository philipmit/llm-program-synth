from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Train the logistic regression model
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_train, y_train)
# Function to predict probabilities for a single sample
def predict_label(sample):
    if len(sample) != X_train.shape[1]:
        raise ValueError("Input data does not have the right shape. It should be ({},), but is {}.".format(X_train.shape[1], sample.shape))
    return log_reg.predict_proba(sample.reshape(1, -1))[0]