from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Load the dataset
wine = load_wine()
X = wine.data
y = wine.target
# Preprocess the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Create a logistic regression model
model = LogisticRegression(multi_class='ovr', solver='liblinear')
# Fit the model to the data
model.fit(X_train, y_train)
def predict_label(raw_data):
    # Preprocess the raw_data
    processed_data = scaler.transform([raw_data])
    # Make a prediction
    prediction = model.predict_proba(processed_data)
    return prediction[0]