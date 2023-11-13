from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
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
# Replace the logistic regression model with a random forest classifier
# Increasing the number of estimators and the depth of trees can enhance performance
model = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
model.fit(X_train, y_train)
# define the function
def predict_icu_mortality(raw_data): 
    raw_data = sc.transform([raw_data]) # Expecting 2D array-like input
    return model.predict_proba(raw_data)[0][1]