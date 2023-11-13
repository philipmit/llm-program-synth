from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Create a new StandardScaler object
sc = StandardScaler()
# Fit the StandardScaler to the training data and transform it
X_train_scaled = sc.fit_transform(X_train)
# Train a logistic regression model on the training data
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
# Define a function to predict label for a single sample
def predict_label(input_data):
        # Scale the input data using the same scaler as before
        input_data_scaled = sc.transform([input_data])
        # Use the logistic regression model to predict probabilities
        probabilities = log_reg.predict_proba(input_data_scaled)
        return probabilities