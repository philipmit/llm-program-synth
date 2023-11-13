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
# Utilize Logistic Regression which is the appropriate model for this problem
# Use 'liblinear' solver for smaller datasets and L1 penalty for feature selection
# Set C to a smaller value for stronger regularization to prevent overfitting
model = LogisticRegression(penalty='l1', C=0.01, solver='liblinear')
model.fit(X_train, y_train)
# define the function
def predict_icu_mortality(raw_data): 
    raw_data = sc.transform([raw_data]) # Expecting 2D array-like input
    return model.predict_proba(raw_data)[0][1]