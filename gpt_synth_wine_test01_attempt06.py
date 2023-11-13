from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize the features to have mean=0 and variance=1
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
# Train the logistic regression model
model = LogisticRegression(multi_class='ovr', solver='liblinear')
model.fit(X_train, y_train)
def predict_label(sample):
    # Preprocess the sample
    sample = scaler.transform([sample])
    # Predict the probabilities of each class
    probabilities = model.predict_proba(sample)
    return probabilities[0]