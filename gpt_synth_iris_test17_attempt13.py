from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train the model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)
def predict_label(raw_data):
    """
    Returns the predicted probabilities of a raw data sample.
    """
    # Ensure the input data is 2D
    raw_data = raw_data.reshape(1, -1)
    # Standardize the raw data
    standardized_data = scaler.transform(raw_data)
    # Predict the probabilities
    predicted_probabilities = model.predict_proba(standardized_data)
    return predicted_probabilities