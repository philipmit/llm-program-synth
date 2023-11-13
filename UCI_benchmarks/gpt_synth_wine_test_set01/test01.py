from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# It's a good practice to scale features before training Logistic Regression model
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)
def predict_label(raw_sample):
    # Standardize the raw_sample to match the data model was trained on
    raw_sample = sc.transform(raw_sample.reshape(1, -1))
    # Return the class probabilities as a 1D array
    return model.predict_proba(raw_sample)[0]  