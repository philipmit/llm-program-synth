from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
def train_model():
    # Load the wine dataset
    wine = load_wine()
    X = wine.data
    y = wine.target
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    # training logistic regression model
    model = LogisticRegression(multi_class="ovr", solver="liblinear")
    model.fit(X_train, y_train)
    return model, scaler
def predict_label(raw_data, model, scaler):
    # preprocessing raw_data using trained scaler
    processed_data = scaler.transform([raw_data])
    # predict the probabilities
    predicts = model.predict_proba(processed_data)
    return predicts
# train the model
model, scaler = train_model()
# prediction example (replace with actual data when available)
raw_data = load_wine().data[0, :]  
prediction = predict_label(raw_data, model, scaler)
print(prediction)