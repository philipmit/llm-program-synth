from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler, RobustScaler
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
# Split the dataset into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Prepare a pipeline with RobustScaler (more robust to outliers compared to StandardScaler) and LogisticRegression
pipeline = make_pipeline(RobustScaler(), LogisticRegression(max_iter=2000))
# Create a dictionary of hyperparameters to search
# Here we increased the range of regularization parameter C exploration and we also included several solvers
grid_param = {"logisticregression__C": np.logspace(-5, 5, 50), 
              "logisticregression__penalty": ["l1", "l2","elasticnet"],
              "logisticregression__solver":["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
              "logisticregression__l1_ratio": np.linspace(0,1,20)}
# Define AUC scorer
scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)
# Initialize GridSearchCV object with the hyperparameters to search, cross-validation increased to 10, and scoring defined as AUC
# GridSearchCV systematically works through multiple combinations of parameter tunes 
gscv = GridSearchCV(pipeline, grid_param, cv=10, scoring=scorer, n_jobs=-1)
# Fit the GridSearchCV object to the data
gscv.fit(X_train, y_train)
# Best model selected
best_logistic_model = gscv.best_estimator_
# Use a Random Forest Classifier to harness the power of decision trees
rfc = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42)
rfc.fit(X_train, y_train)
# The following function will be used to predict the labels using the trained models above 
# The final prediction will be a simple average of the two model predictions
def predict_icu_mortality(raw_sample):
    # predict probability for a single sample
    single_sample_reshape = raw_sample.reshape(1, -1)
    logistic_prediction_prob = best_logistic_model.predict_proba(single_sample_reshape)[0, 1]
    rfc_prediction_prob = rfc.predict_proba(single_sample_reshape)[0, 1]
    # We use average of both predictions
    aggregated_prediction = (logistic_prediction_prob + rfc_prediction_prob) / 2
    return aggregated_prediction