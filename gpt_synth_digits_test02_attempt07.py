from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize the features to have mean=0 and variance=1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Initialize a Logistic Regression model
log_reg = LogisticRegression(max_iter=5000)
# Train the model
log_reg.fit(X_train, y_train)
def predict_label(single_sample):
    # The input is assumed to be a raw, unprocessed data row
    # We have to reshape it to a 2D array and apply the feature scaling
    single_sample = scaler.transform(single_sample.reshape(1, -1))
    # Return the model's prediction
    return log_reg.predict_proba(single_sample)