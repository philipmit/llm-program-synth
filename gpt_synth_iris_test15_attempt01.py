from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Initialize the StandardScaler
sc = StandardScaler()
# Fit the StandardScaler with the training data and transform the training data
X_train = sc.fit_transform(X_train)
# Fit a Logistic Regression model with the training data
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
# Define the function to predict labels
def predict_label(single_sample):
    # Apply the same scaling to the single_sample that was applied to the training data
    single_sample = sc.transform([single_sample])
    # Use the trained model to make predictions
    probabilities = model.predict_proba(single_sample)
    # Since we are only predicting one instance, we just need to return the first element
    return probabilities[0]