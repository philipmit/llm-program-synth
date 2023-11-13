from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardizing the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Training the logistic regression model
logisticRegr = LogisticRegression(solver = 'lbfgs', max_iter = 5000, multi_class ='multinomial')
logisticRegr.fit(X_train, y_train)
def predict_label(raw_data):
    # Preprocess raw data
    raw_data = sc.transform(raw_data.reshape(1, -1))
    # Perform prediction
    predicted_probabilities = logisticRegr.predict_proba(raw_data)
    return predicted_probabilities