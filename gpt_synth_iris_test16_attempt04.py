from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Scale the features to have mean=0 and variance=1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
# function to predict label for new data
def predict_label(new_data):
    # preprocess the new_data like the training data
    new_data = scaler.transform(new_data.reshape(1, -1))
    # predict and return the probabilities
    return model.predict_proba(new_data)