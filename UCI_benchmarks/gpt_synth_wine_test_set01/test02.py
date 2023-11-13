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
# Scale the features to have zero mean and unit variance using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train logistic regression model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
def predict_label(sample):
    """
    This function takes raw unprocessed data for one sample and returns predicted probabilities 
    after preprocessing and applying the model.
    """
    # Preprocess (scale) the sample and reshape it for prediction
    sample = scaler.transform(sample.reshape(1, -1))
    # Compute and return probabilities
    probabilities = model.predict_proba(sample)
    return probabilities