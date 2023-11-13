from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Train the Logistic Regression model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
def predict_label(raw_sample):
    # predict label's probability for a single sample
    single_sample_reshape = raw_sample.reshape(1, -1)
    prob = model.predict_proba(single_sample_reshape)[0, 1]
    return prob