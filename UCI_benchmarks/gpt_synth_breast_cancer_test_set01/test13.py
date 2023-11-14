from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler, RobustScaler
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
import numpy as np
# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Prepare a pipeline with RobustScaler (more robust to outliers compared to StandardScaler) and LogisticRegression
pipeline = make_pipeline(RobustScaler(), LogisticRegression(max_iter=1000))
# Create a dictionary of hyperparameters to search
# C: A smaller value of C leads to more regularization, while a larger value leads to less regularization
grid_param = {"logisticregression__C": np.logspace(-5, 5, 15), 
              "logisticregression__penalty": ["l1", "l2","elasticnet"],
              "logisticregression__solver":["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
              "logisticregression__l1_ratio": np.linspace(0,1,10)}
# Define AUC scorer
scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)
# Initialize GridSearchCV object with the hyperparameters to search, cross-validation of 5, and scoring defined as AUC
# GridSearchCV systematically works through multiple combinations of parameter tunes 
gscv = GridSearchCV(pipeline, grid_param, cv=5, scoring=scorer, n_jobs=-1)
# Fit the GridSearchCV object to the data
gscv.fit(X_train, y_train)
# Best model selected
best_logistic_model = gscv.best_estimator_
# Now we train Gradient Boosting Classifier
gbm = GradientBoostingClassifier(n_estimators=200,
                                 learning_rate=0.1, 
                                 subsample=.8, 
                                 max_features=.3,
                                 random_state=42)
gbm.fit(X_train,y_train)
# Also, we train the AdaBoost Classifier
abc = AdaBoostClassifier(n_estimators=200, random_state=42)
abc.fit(X_train,y_train)
# The following function will be used to predict the labels of new data using the trained models above 
# The final prediction will be a majority vote of the three models
def predict_icu_mortality(raw_sample):
    # predict probability for a single sample
    single_sample_reshape = raw_sample.reshape(1, -1)
    logistic_prediction_prob = best_logistic_model.predict_proba(single_sample_reshape)[0, 1]
    gbm_prediction_prob = gbm.predict_proba(single_sample_reshape)[0, 1]
    abc_prediction_prob = abc.predict_proba(single_sample_reshape)[0, 1]
    # We use average of all three predictions here
    aggregated_prediction = (logistic_prediction_prob + gbm_prediction_prob + abc_prediction_prob) / 3
    return aggregated_prediction