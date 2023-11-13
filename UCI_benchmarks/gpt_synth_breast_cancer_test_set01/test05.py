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
# Normalize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
# Define the predict_label function
def predict_label(sample):
    # Normalize the sample
    sample_scaled = scaler.transform([sample])
    # Predict the probability of the label being 1
    return model.predict_proba(sample_scaled)[0][1]