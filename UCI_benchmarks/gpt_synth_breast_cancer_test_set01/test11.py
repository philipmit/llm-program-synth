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
X = sc.fit_transform(X) 
# Split the dataset into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42) 
# Train a logistic regression model with L1 penalty and saga solver 
model = LogisticRegression(penalty='l1', solver='saga', max_iter=10000, C=1.0) 
model.fit(X_train, y_train) 
def predict_icu_mortality(raw_data): 
    raw_data = sc.transform([raw_data]) 
    return model.predict_proba(raw_data)[0][1] 