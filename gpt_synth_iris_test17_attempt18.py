from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
def predict_label(sample):
    sample = sample.reshape(1, -1)
    probabilities = model.predict_proba(sample)
    return probabilities