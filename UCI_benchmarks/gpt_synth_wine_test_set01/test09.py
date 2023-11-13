from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
def predict_label(sample):
    # Preprocess the sample as per model requirements
    sample = sample.reshape(1, -1)
    sample = scaler.transform(sample)
    prob = model.predict_proba(sample)
    return prob