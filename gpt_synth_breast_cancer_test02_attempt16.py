from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42)
# Standardize the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
# Train the Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)
def predict_label(raw_sample):
    # preprocessing raw_sample
    processed_sample = sc.transform([raw_sample])
    # predicting probability
    prob = lr.predict_proba(processed_sample)[0][1]
    return prob