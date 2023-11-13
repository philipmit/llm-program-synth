from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
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
# Define a Random Forest model, an ensemble method which uses multiple decision trees to improve the performance
model = RandomForestClassifier(n_estimators=100, random_state=42)
# Define the grid of hyperparameters to search
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
# Set up the grid search with 3-fold cross validation
grid_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv= 3)
grid_cv.fit(X_train, y_train)
# Train the optimized model
model = grid_cv.best_estimator_
model.fit(X_train, y_train)
# Define the function
def predict_icu_mortality(raw_data): 
    raw_data = sc.transform([raw_data]) 
    return model.predict_proba(raw_data)[0][1]