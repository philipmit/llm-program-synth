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
# Data standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Logistic regression model
model = LogisticRegression(multi_class='ovr', solver='liblinear')
model.fit(X_train, y_train)
def predict_label(in_data_raw):
    in_data = scaler.transform([in_data_raw])
    prob = model.predict_proba(in_data)
    return prob[0]