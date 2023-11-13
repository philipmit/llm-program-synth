from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Standardize the features for better prediction
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
# Train the Logistic Regression model
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)
def predict_label(raw_data):
    processed_data = sc.transform(raw_data.reshape(1, -1))
    probabilities = clf.predict_proba(processed_data)
    return probabilities[0]  # return only the first element (1D) instead of 2D array