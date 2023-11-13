from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
# Standardize the features to have mean=0 and variance=1
sc = StandardScaler()
X = sc.fit_transform(X)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Initialize and fit the logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
# Define the predict function
def predict_label(input_data):
    # Make sure the input data is a 2D array
    input_data = input_data.reshape(1, -1)
    # Standardize the input data
    input_data = sc.transform(input_data)
    # Predict the probability of the label being 1
    predicted_proba = log_reg.predict_proba(input_data)
    # Return the probability of the label being 1
    return predicted_proba[0][1]