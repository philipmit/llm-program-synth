import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
# Load the ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Prepare the data 
X = ecoli.iloc[:, 1:-1]
y = ecoli.iloc[:, -1]
# Transform y from categorical to numerical values
le = LabelEncoder()
y = le.fit_transform(y)
# Stratified split of data into training and test sets, ensuring all classes are represented
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Define and train the model
param_grid = {
    'gb__learning_rate': [0.01, 0.1, 0.5, 1],
    'gb__n_estimators': [100, 200, 500],
    'gb__max_depth': [3, 5, 10]
}
gb = GradientBoostingClassifier(random_state=42)
pipe = Pipeline(steps=[('standardize', StandardScaler()), ('gb', gb)])
clf = GridSearchCV(pipe, param_grid, cv=5)
clf.fit(X_train, y_train)
# Get the best model
model = clf.best_estimator_
# Define the prediction function
def predict_label(raw_data):
    """
    Predict probabilities for a given raw unprocessed data.
    """
    raw_data = np.array(raw_data).reshape(1, -1)
    probabilities = model.predict_proba(raw_data)[0]
    return probabilities