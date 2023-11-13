from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Train a Logistic regression model
clf = LogisticRegression(random_state=42)
clf.fit(X_train,y_train)
def predict_label(raw_data_sample):
    # Standardize the raw data sample
    raw_data_sample_scaled = sc.transform(raw_data_sample.reshape(1, -1))
    # Predict the probabilities
    probabilities = clf.predict_proba(raw_data_sample_scaled)
    return probabilities