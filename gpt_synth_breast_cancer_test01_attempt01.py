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
# Scale the inputs with StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train the Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
def predict_label(X):
    # Scaling the input
    X = scaler.transform([X])
    # Predicting probabilities for both classes
    probabilities = model.predict_proba(X)
    # Return probabilities for both classes
    return probabilities[0]