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
# It's a good practice to scale the features so that all of them can be uniformly evaluated.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Initialize the logistic regression model
lr = LogisticRegression(multi_class='ovr', solver='liblinear')
# Train the model
lr.fit(X_train, y_train)
def predict_label(raw_data):
    # Scale the raw data using the same scaler used for training
    raw_data_scaled = scaler.transform(raw_data.reshape(1, -1))
    # Make prediction
    prediction = lr.predict_proba(raw_data_scaled)
    return prediction