from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
def train_model():
    # Load the wine dataset
    wine = load_wine()
    X = wine.data
    y = wine.target
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    # Normalization of data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    # Logistic regression model
    model = LogisticRegression(multi_class='ovr', solver='liblinear')
    model.fit(X_train, y_train)
    return model, scaler
def predict_label(model, scaler, in_data_raw):
    # Preprocessing on raw input
    in_data = scaler.transform([in_data_raw])
    # Predict the output for the sample
    prob = model.predict_proba(in_data)
    return prob
model, scaler = train_model()