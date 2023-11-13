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
# Initialize a StandardScaler and fit it to the training data
scaler = StandardScaler().fit(X_train)
# Transform X_train using the trained scaler
X_train = scaler.transform(X_train)
# Initialize a Logistic Regression model
log_reg = LogisticRegression(random_state=42)
# Fit the model to the training data
log_reg.fit(X_train, y_train)
def predict_label(raw_data):
    # Preprocess the raw data using the trained scaler
    preprocessed_data = scaler.transform(raw_data.reshape(1, -1))
    # Predict the probability using the trained model
    prob = log_reg.predict_proba(preprocessed_data)
    # Return the probability of label being 1
    return prob[0][1]