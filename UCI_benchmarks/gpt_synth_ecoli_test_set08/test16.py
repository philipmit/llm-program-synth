import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np
class LogisticModel:
    def __init__(self):
        self.model = LogisticRegression()
        self.le = LabelEncoder()
        self.fitted = False
    def load_data(self):
        ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
        ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
        y = ecoli.iloc[:, -1]
        self.le.fit(y)
        y = self.le.transform(y)
        X = ecoli.iloc[:, 1:-1].to_numpy()
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        return X_train, y_train
    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.fitted = True
    def predict_label(self, raw_input):
        assert self.fitted, "Model not yet fitted"
        probabilities = self.model.predict_proba(raw_input.reshape(1, -1))
        return probabilities[0]
# create an instance of the model and train it
my_model = LogisticModel()
X_train, y_train = my_model.load_data()
my_model.train_model(X_train, y_train)
# redefine predict_label to use the instance of LogisticModel - assuming that the third argument is no longer required
def predict_label(raw_input):
    return my_model.predict_label(raw_input)