from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, roc_auc_score
import numpy as np
# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Prepare a pipeline with StandardScaler and LogisticRegression
pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500000))
# Create a dictionary of hyperparameters to search
# C: A smaller value of C leads to more regularization, while a larger value leads to less regularization
grid_param = {"logisticregression__C": np.logspace(-3, 3, 7), 
              "logisticregression__penalty": ["l1", "l2"],
              "logisticregression__solver":["newton-cg", "lbfgs", "liblinear", "sag", "saga"]}
# Initializing AUC scorer
scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)
# Initialize RandomizedSearchCV object with the hyperparameters to search and cross-validation setting
rscv = RandomizedSearchCV(pipeline, grid_param, cv=5, scoring=scorer, n_iter=100, random_state=42, 
                          n_jobs=-1)
# Fit the RandomizedSearchCV object to the data
rscv.fit(X_train, y_train)
# Best model selected
best_logistic_model = rscv.best_estimator_
# Similarly, we train a SVM and Random Forest classifier
svm = RandomizedSearchCV(make_pipeline(StandardScaler(), SVC(probability=True)), {
    'svc__C': np.logspace(-3, 3, 7),
    'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}, cv=5, scoring=scorer, n_iter=100, random_state=42, n_jobs=-1)
svm.fit(X_train, y_train)
best_svm_model = svm.best_estimator_
rf = RandomizedSearchCV(RandomForestClassifier(), {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10]
}, cv=5, scoring=scorer, n_iter=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
best_rf_model = rf.best_estimator_
# Ensemble the three models
ensemble = VotingClassifier(estimators=[('lr', best_logistic_model), ('svm', best_svm_model), ('rf', best_rf_model)], voting='soft')
ensemble.fit(X_train, y_train)
def predict_icu_mortality(raw_sample):
    # predict label's probability for a single sample
    single_sample_reshape = raw_sample.reshape(1, -1)
    prob = ensemble.predict_proba(single_sample_reshape)[0, 1]
    return prob