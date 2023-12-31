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
# Standardize the features for more accurate results
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Train the logistic regression model
model = LogisticRegression(multi_class='ovr', solver='liblinear')
model.fit(X_train_scaled, y_train)
# Function to predict the label of a single sample
def predict_label(sample):
    sample_scaled = scaler.transform(sample.reshape(1, -1))  # Standardize and reshape the sample
    probabilities = model.predict_proba(sample_scaled)  # Predict the probabilities
    return probabilities[0] # Return the only item