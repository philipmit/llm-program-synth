import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
def predict_label(sample):
    sample = np.array(sample).reshape(1, -1)
    sample = preprocess_pipeline.transform(sample)
    predicted_probs = model.predict_proba(sample)
    return predicted_probs[0]
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]
y = ecoli.iloc[:, -1]
unique_y = np.unique(y)
y = y.replace(unique_y, np.arange(len(unique_y)))
X = X.to_numpy()
y = y.to_numpy()
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
# Use a more complex pipeline which includes feature selection based on model importance
preprocess_pipeline = make_pipeline(StandardScaler(), SelectFromModel(estimator=RandomForestClassifier(n_estimators=100, random_state=42)), PCA(n_components=0.95))
X_train = preprocess_pipeline.fit_transform(X_train)
# Use a more complex model, random forest classifier which is typically better performing than logistic regression
model = RandomForestClassifier(n_estimators=500, criterion='entropy', max_features='sqrt', oob_score=True, random_state=42)
model.fit(X_train, y_train)