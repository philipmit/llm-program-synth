from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Standardization of the dataset for improved performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train the Logistic Regression model
model = LogisticRegression(multi_class='ovr', solver='liblinear')
model.fit(X_train, y_train)
def predict_label(raw_data):
    processed_data = scaler.transform([raw_data])
    predicted_probabilities = model.predict_proba(processed_data)
    return predicted_probabilities