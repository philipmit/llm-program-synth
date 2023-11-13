from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Preprocess the inputs
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)
# Define a function that takes a raw sample and predicts the class using the trained model
def predict_label(X):
    X = scaler.transform([X])
    predictions = model.predict_proba(X)
    # Return the prediction probability for the positive class
    return predictions[0][1]