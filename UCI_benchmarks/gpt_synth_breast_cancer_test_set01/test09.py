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
# Standardize the features to have mean=0 and variance=1
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
# Initialize and train logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)
def predict_label(raw_data):
    # Reshape the data and standardize it
    processed_data = scaler.transform(raw_data.reshape(1, -1))
    # Return the probability of the label being 1
    return log_reg.predict_proba(processed_data)[0][1]