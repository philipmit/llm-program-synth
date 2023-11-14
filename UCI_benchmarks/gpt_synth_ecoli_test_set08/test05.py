import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1].to_numpy()  # All rows, all columns except the last one
y = ecoli.iloc[:, -1].replace(list(np.unique(ecoli.iloc[:, -1])), [0,1,2,3,4,5,6,7]).to_numpy()   # All rows, only the last column which is replaced with integers
# Standardizing the X
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Use StratifiedShuffleSplit to make sure all classes are represented in the train/test split
strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in strat_split.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
# Training the model
model = LogisticRegression(multi_class='multinomial', max_iter=5000, solver='saga', penalty='l1', C=0.5)
model.fit(X_train, y_train)
# Define the function predict_label
def predict_label(sample):
    sample = scaler.transform([sample]) # Standardize the new inputs as well
    return model.predict_proba(sample)[0]