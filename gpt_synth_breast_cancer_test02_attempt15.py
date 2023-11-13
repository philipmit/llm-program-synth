from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Data Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Define the logistic regression model
model = LogisticRegression()
# Train the model
model.fit(X_train_scaled, y_train)
# Function to predict label
def predict_label(raw_data):
    processed_data = scaler.transform(raw_data.reshape(1, -1))
    predicted_prob = model.predict_proba(processed_data)[:, 1]
    return predicted_prob[0]