import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, 
header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.drop(['Sequence Name', 'class'], axis=1)  
y = ecoli['class']  
# Replace strings with numbers in y
class_labels = np.unique(y)
label_dict = {label: index for index, label in enumerate(class_labels)}
y = y.replace(label_dict)
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.4, random_state=42)
# Normalize the dataset and add feature selection in the pipeline
pipe = Pipeline([
  ('scaler', StandardScaler()), 
  ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100,random_state=42))),
  ('classification', LogisticRegression(C=0.9,max_iter=5000, multi_class='multinomial',solver='lbfgs'))])
pipe.fit(X_train,y_train)
def predict_icu_mortality(raw_sample):
    """Predicts the mortality in ICU of a single sample using the trained model."""
    raw_sample = np.array(raw_sample).reshape(1, -1)
    probas = pipe.predict_proba(raw_sample)
    return probas[0]