from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Training the logistic regression model
logisticRegr = LogisticRegression(solver = 'lbfgs', max_iter = 5000, multi_class ='multinomial')
logisticRegr.fit(X_train, y_train)
# Define predict_label function
def predict_label(raw_data):
    # Preprocess raw data
    data = scaler.transform(raw_data.reshape(1, -1))
    # Return predicted probabilities
    return logisticRegr.predict_proba(data)