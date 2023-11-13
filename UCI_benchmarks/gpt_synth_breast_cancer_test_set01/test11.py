from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
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
# Define a logistic regression model. 
model = LogisticRegression(max_iter=1000, tol=0.1)
# Create a list of options for the regularization penalty
penalty = ['l1', 'l2', 'elasticnet', 'none']
# Create a list of options for the solver
solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
# Define the grid of hyperparameters to search
hyperparameter_grid = {'penalty': penalty, 'solver': solver}
# Set up the grid search 
grid_cv = GridSearchCV(estimator=model, 
                       param_grid=hyperparameter_grid,
                       cv=5, 
                       n_jobs =-1, 
                       scoring = 'roc_auc')
# Conduct the grid search 
grid_cv.fit(X_train, y_train)
# Find out the best hyperparameters
print('Best Penalty:', grid_cv.best_estimator_.get_params()['penalty'])
print('Best solver:', grid_cv.best_estimator_.get_params()['solver'])
# Train the optimized logistic regression model
model = grid_cv.best_estimator_
model.fit(X_train, y_train)
def predict_icu_mortality(raw_data): 
    raw_data = sc.transform([raw_data]) 
    return model.predict_proba(raw_data)[0][1]