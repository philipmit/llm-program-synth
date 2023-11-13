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
# Perform feature scaling on the training set
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
# Define a function that makes predictions on unprocessed data
def predict_label(raw_data):
    raw_data = raw_data.reshape(1, -1) # reshape to 1 sample 
    processed_data = scaler.transform(raw_data) # apply the same transformations used on the training data
    probabilities = model.predict_proba(processed_data)[0] # get probabilities and extract the list (not nested)
    return probabilities