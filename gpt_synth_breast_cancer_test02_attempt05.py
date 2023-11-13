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
# Standardize the features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
# Define the function to predict the label for a new sample
def predict_label(raw_data):
    # Preprocess the raw data
    raw_data = scaler.transform(raw_data.reshape(1, -1))
    # Return the probability of the label being 1
    return model.predict_proba(raw_data)[0][1]