from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, roc_auc_score
# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Prepare a pipeline with StandardScaler and LogisticRegression
pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
# Create a dictionary of hyperparameters to search
# C: A smaller value of C leads to more regularization, while a larger value leads to less regularization
grid_param = {"logisticregression__C": [0.01, 0.1, 1, 10, 100], "logisticregression__solver":["newton-cg", "lbfgs", "liblinear"]}
# Initializing AUC scorer
scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)
# Initialize GridSearchCV object with the hyperparameters to search and cross-validation setting
gcv = GridSearchCV(pipeline, grid_param, cv=5, scoring=scorer)
# Fit the GridSearchCV object to the data
gcv.fit(X_train, y_train)
# Best model selected
best_model = gcv.best_estimator_
def predict_icu_mortality(raw_sample):
    # predict label's probability for a single sample
    single_sample_reshape = raw_sample.reshape(1, -1)
    prob = best_model.predict_proba(single_sample_reshape)[0, 1]
    return prob