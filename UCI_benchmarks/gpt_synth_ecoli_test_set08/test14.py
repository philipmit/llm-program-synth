import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score
# Load the ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Prepare the data 
X = ecoli.iloc[:, 1:-1]  
y = ecoli.iloc[:, -1]
# Transform y from categorical to numerical values
le = LabelEncoder()
y = le.fit_transform(y)
# Stratified Split of data into Training and Test set ensuring all classes are represented
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
# Training a logistic regression model with feature selection and data standardization
model = Pipeline([
  ('feature_selection', RFECV(LogisticRegression(max_iter=1000, multi_class='ovr'))),
  ('standardize', StandardScaler()),
  ('log_reg', LogisticRegression(max_iter=5000, multi_class='ovr', solver='saga', penalty='l1', C=0.1))
])
model.fit(X_train, y_train)
def predict_label(raw_data):
    """ 
    Predict probabilities for a given raw unprocessed data. 
    """
    raw_data = np.array(raw_data).reshape(1, -1)
    probabilities = model.predict_proba(raw_data)
    return probabilities[0]