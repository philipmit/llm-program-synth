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
# Apply Standard Scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Training logistic regression model
model = LogisticRegression(multi_class="ovr", solver="liblinear")
model.fit(X_train, y_train)
def predict_label(raw_data):
    # Apply preprocessing using trained scaler
    processed_data = scaler.transform([raw_data])
    # Predict the probabilities
    predicts = model.predict_proba(processed_data)
    return predicts[0]