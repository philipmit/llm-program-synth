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
# Standardizing the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
# Train logistic regression model
model = LogisticRegression(multi_class='ovr', solver='liblinear')
model.fit(X_train, y_train)
def predict_label(raw_data):
    # Ensure the raw_data is 2D
    raw_data = np.array(raw_data).reshape(1, -1)
    # Apply the same scaling to the raw_data as was applied to X_train
    raw_data = sc.transform(raw_data)
    # Use the logistic regression model to predict the probabilities
    pred = model.predict_proba(raw_data)
    return pred