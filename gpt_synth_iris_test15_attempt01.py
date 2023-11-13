from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Standardize the features to have a mean=0 and variance=1
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
# Train the Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
def predict_label(single_sample):
    # Apply the same scaling to the single_sample
    single_sample = sc.transform(single_sample.reshape(1, -1))
    # Predict probabilities
    probabilities = model.predict_proba(single_sample)
    return probabilities