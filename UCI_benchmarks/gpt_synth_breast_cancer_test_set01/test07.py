from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
features = breast_cancer.feature_names
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize the data (mean=0, std=1) using training set
scaler = StandardScaler().fit(X_train)
# Apply the standardization to both training and testing sets
X_train_std = scaler.transform(X_train)
# Create a Logistic Regression classifier and train it
clf = LogisticRegression(random_state=42).fit(X_train_std, y_train)
def predict_label(raw_data):
    # Reshape the input to a 2D array 
    raw_data = raw_data.reshape(1, -1)
    # Standardize the raw data
    raw_data_std = scaler.transform(raw_data)
    # Predict the probability and return the probability for label 1
    return clf.predict_proba(raw_data_std)[0][1] 