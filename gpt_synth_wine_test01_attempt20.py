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
# Standardize the features for better performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Define and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
# Define the prediction function
def predict_label(raw_data):
    # Preprocess the input using same scaler
    processed_data = scaler.transform([raw_data])
    # Predict probabilities
    predicted_probs = model.predict_proba(processed_data)
    return predicted_probs