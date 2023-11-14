import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
class LogisticModel:
    def __init__(self):
        # Using 'saga' solver, L1 penalty for potential better performance
        # And increasing max_iter to give the optimizer more space to reach to optimal solution
        self.model = LogisticRegression(solver='saga', penalty='l1', max_iter=5000, random_state=42)
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
        X = self.scaler.fit_transform(X) # Standard scale the input for better convergence in LR
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
my_model = LogisticModel()
X_train, y_train = my_model.load_data()
my_model.train_model(X_train, y_train)
# redefine predict_label to use the instance of LogisticModel - assuming that the third argument is no longer required
def predict_label(raw_input):
    return my_model.predict_label(raw_input)