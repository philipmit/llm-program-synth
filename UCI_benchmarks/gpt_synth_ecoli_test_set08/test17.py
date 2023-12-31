import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# Apply label encoding to the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y.astype(str))
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Using Gradient Boosting Classifier instead of Random Forest Classifier for better performance
gbm = GradientBoostingClassifier(n_estimators=500, learning_rate=0.01, random_state=42)
# Create a pipeline that scales the data and then trains a Gradient Boosting model
pipeline = make_pipeline(
    StandardScaler(),
    gbm
)
# Use GridSearchCV to find the best hyperparameters
param_grid = {
    'gradientboostingclassifier__max_depth': [3, 5, 10],
    'gradientboostingclassifier__min_samples_split': [2, 5, 10],
    'gradientboostingclassifier__subsample': [0.5, 0.7, 1.0]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)
# Get the best model
model = grid_search.best_estimator_
# Define the predict_label function
def predict_label(raw_sample):
    # Preprocess the input
    sample = np.array(raw_sample).reshape(1, -1)
    # Use the trained Gradient Boosting model to predict the probabilities
    predicted_probabilities = model.predict_proba(sample)
    # Adding probabilities for missing classes
    full_proba = np.zeros(len(label_encoder.classes_))
    for class_idx, proba in zip(model.classes_, predicted_probabilities[0]):
        full_proba[class_idx] = proba
    return full_proba