from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Define and fit the logistic regression model
lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train)
def predict_label(raw_data):
    processed_data = scaler.transform([raw_data]) 
    probabilities = lr.predict_proba(processed_data)
    return probabilities[0]