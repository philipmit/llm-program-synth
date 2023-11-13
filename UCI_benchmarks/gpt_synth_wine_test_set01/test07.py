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
# Standardize the Features Matrix
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Create and train the logistic regression model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
# Prediction function
def predict_label(sample):
    # Preprocess the sample
    sample = scaler.transform([sample])
    # Use the model to predict the labels
    predictions = model.predict_proba(sample)
    return predictions