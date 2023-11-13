from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Initialize our logistic regression model
model = LogisticRegression()
# Train the model
model.fit(X_train, y_train)
# Function to predict the labels for a single sample
def predict_label(raw_sample):
    # Reshaping the raw_sample into shape (1, -1) because sklearn expects 2D array as input
    reshaped_sample = raw_sample.reshape(1, -1)
    return model.predict_proba(reshaped_sample)