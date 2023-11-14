import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Download dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', 
                    delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Preprocessing
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]  # All rows, only the last column
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7])
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Create a StandardScaler instance and fit it to the training data
scaler_all = StandardScaler()
scaler_all.fit(X)  
X_train = scaler_all.transform(X_train)
# Fit logistic regression to the training set
logisticRegr = LogisticRegression(max_iter=1000)
logisticRegr.fit(X_train, y_train)
def predict_label(sample):
    sample = scaler_all.transform([sample])  # apply the same scaler used in training to the new sample
    return logisticRegr.predict_proba(sample)[0]  # return probabilities for each class