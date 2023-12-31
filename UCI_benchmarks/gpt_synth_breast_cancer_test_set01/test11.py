from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the Breast Cancer dataset 
breast_cancer = load_breast_cancer() 
X = breast_cancer.data 
y = breast_cancer.target 
# Feature Scaling 
sc = StandardScaler() 
X_sc = sc.fit_transform(X)
# Split the dataset into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X_sc, y, test_size=0.5, random_state=42)
# Use logistic regression model with 'liblinear' solver which works quite well for small dataset and binary classification like this case.
# Increasing the number of iterations can improve the convergence of the model (but it can also lead to overfitting)
model = LogisticRegression(solver="liblinear", max_iter=1000, random_state=0)
model.fit(X_train, y_train)
# Define the function
def predict_label(raw_data): 
    raw_data = sc.transform([raw_data]) # Expecting 2D array-like input
    return model.predict_proba(raw_data)[0][1]