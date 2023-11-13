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
# Standardize the features to speed up the Logistic Regression
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)
# Function to predict the label of a single sample
def predict_label(sample):
    sample = sc.transform([sample])  # Applying the same standardization to the sample
    return model.predict_proba(sample)[0][1]  # Return the probability of belonging to positive class