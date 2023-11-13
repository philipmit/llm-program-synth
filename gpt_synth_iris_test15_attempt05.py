from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Train the logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
def predict_label(unprocessed_sample):
    return model.predict_proba([unprocessed_sample])[0]