import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
class GradientBoostingModel:
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=1000, 
                                                learning_rate=0.01, 
                                                max_depth=5, 
                                                random_state=42)
        self.le = LabelEncoder()
        self.scaler = StandardScaler()
        self.fitted = False
    def load_data(self):
        ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
        ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
        y = ecoli.iloc[:, -1]
        self.le.fit(np.unique(y))  # ensure label encoder is fitted with all existing classes
        y = self.le.transform(y)
        X = ecoli.iloc[:, 1:-1].to_numpy()
        self.scaler.fit(X)  # fit scaler for later use in prediction
        X = self.scaler.transform(X)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        return X_train, y_train
    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.fitted = True
    def predict_label(self, raw_input):
        assert self.fitted, "Model not yet fitted"
        raw_input = self.scaler.transform(raw_input.reshape(1, -1))
        probabilities = self.model.predict_proba(raw_input)
        # ensure the returned probabilities have probabilities for all classes
        return np.append(probabilities, [0]*(len(self.le.classes_) - len(probabilities[0])))
# create an instance of the model and train it
my_model = GradientBoostingModel()
X_train, y_train = my_model.load_data()
my_model.train_model(X_train, y_train)
# redefine predict_label to use the instance of GradientBoostedModel
def predict_label(raw_input):
    return my_model.predict_label(raw_input)