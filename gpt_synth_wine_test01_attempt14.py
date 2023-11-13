from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize the data features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
def predict_label(raw_sample):
    # Standardize the raw data sample
    processed_sample = scaler.transform([raw_sample])[0]
    # Use the model to predict the probabilities
    predicted_probabilities = model.predict_proba([processed_sample])
    return predicted_probabilities