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
# Scale the features to have zero mean and unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
def predict_label(sample):
    """
    This function takes raw unprocessed data for a single sample
    and returns the predicted probabilities after preprocessing and applying the model.
    Parameters:
    sample (array-like of shape (n_features,)): Individual sample with n_features from the wine dataset
    Returns:
    probabilities (ndarray of shape (n_classes,)): Predicted probabilities for each class
    """
    # Preprocess (scale) the sample and reshape it for prediction
    sample = scaler.transform(sample.reshape(1, -1))
    # Compute and return probabilities
    probabilities = model.predict_proba(sample)
    return probabilities