from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Standardize the features to have mean=0 and variance=1
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
# Define and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
def predict_label(raw_data):
    # Preprocess the raw input data like the training data
    processed_data = sc.transform(raw_data.reshape(1, -1))
    # Get the probability of the positive class
    prob_one = model.predict_proba(processed_data)[0][1]
    return prob_one