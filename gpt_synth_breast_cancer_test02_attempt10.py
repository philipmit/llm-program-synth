from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize the features to have zero mean and unit variance
scaler = StandardScaler().fit(X_train)
# perform transformation
X_train = scaler.transform(X_train)
# Create a logistic regression object
log_reg = LogisticRegression(random_state=42)
# Train the logistic regression object on the training data
log_reg.fit(X_train, y_train)
def predict_label(x):
    # Transform the raw sample with the same scaler used for training
    x = scaler.transform([x])
    # Return the predicted probability for label being 1
    return log_reg.predict_proba(x)[:, 1]