from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target
# Define the standard scaler
scaler = StandardScaler()
# Scale the features
X_scaled = scaler.fit_transform(X)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=42)
# Train the Logistic Regression model
clf = LogisticRegression()
clf.fit(X_train, y_train)
def predict_label(raw_data):
    # Reshaping the raw_data to correctly input into transform and predict methods
    raw_data = raw_data.reshape(1, -1)
    # The raw data needs to be preprocessed (scaled) before being input into the model
    processed_data = scaler.transform(raw_data)
    # Returns the predicted probabilities
    return clf.predict_proba(processed_data)[0]