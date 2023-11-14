from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Initialize and train the logistic regression model
logreg = LogisticRegression(max_iter=10000)
logreg.fit(X_train, y_train)
def predict_label(raw_data):
    prob = logreg.predict_proba([raw_data])
    return prob[0]