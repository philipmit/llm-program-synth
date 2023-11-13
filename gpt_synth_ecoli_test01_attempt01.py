import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# Load the digits dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# Replace string labels with numbers in y using LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # This will assign a unique number to each class label starting from zero
X=X.to_numpy()
y=y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Create a pipeline with StandardScaler and RandomForest Classifier and tune the parameters
pipeline = make_pipeline(StandardScaler(), RandomForestClassifier())
parameters = {'randomforestclassifier__n_estimators': [50, 100, 150, 200],
              'randomforestclassifier__max_depth': [None, 5, 10, 15], 
              'randomforestclassifier__min_samples_split': [2, 4, 6]}
search = GridSearchCV(pipeline, parameters, cv=5)
search.fit(X_train, y_train)
# After tuning parameters, we can access the best model through 'best_estimator_'
best_model = search.best_estimator_
# Defining a function called predict_label that takes one input, the raw unprocessed data for a single sample, and returns one output
def predict_label(raw_sample):
    processed_sample = np.array([raw_sample])  # Process the raw unprocessed sample to match the format of X
    num_classes = len(np.unique(y))  # Get the total number of classes
    proba = best_model.predict_proba(processed_sample)[0]   # Predict probabilities
    # Padding the probabilities array with zeros if probabilities for some classes are missing
    missing_classes = num_classes - len(proba)
    if missing_classes > 0:
        proba = np.pad(proba, (0, missing_classes), 'constant')
    return proba   # Return the probabilities for each class