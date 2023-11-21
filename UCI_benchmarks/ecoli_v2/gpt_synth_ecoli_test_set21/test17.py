#<PrevData>
######## Load and preview the dataset and datatypes
# Import necessary libraries
import pandas as pd
# Read file
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', header=None)
# Preview dataset and datatypes
print(df.shape)
print(df.head())
print(df.info())
print(df.dtypes)
for col in df.applymap(type).columns:
    print(col, df.applymap(type)[col].unique())
print(df.isnull().sum())
#</PrevData>

#<PrepData>
######## Prepare the dataset for training
# Import necessary packages
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  
y = ecoli.iloc[:, -1]  
# replace strings with numbers in y
np.unique( y)
len(list(np.unique( y)))
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7])
X=X.to_numpy()
y=y.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,test_size=0.5, random_state=42)
X_train=X_train.tolist()
X_test=X_test.tolist()
# Scale the features 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
print(X_train.shape)
print(y_train.shape)
print(X_train[0:5])
print(y_train[0:5])
#</PrepData>

#<Train>
######## Train the model using the training data, X_train and y_train
# Import necessary packages
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Create a logistic regression instance
logreg = LogisticRegression()

# Create a one-vs-rest logistic regression instance
multilr = OneVsRestClassifier(logreg)

# Fit the model
multilr.fit(X_train, y_train)
#</Train>
#<Predict>
######## Define the predict_label function that can be used to make new predictions using the trained model above given one sample from X_test
def predict_label(sample):
    sample = np.array(sample).reshape(1,-1)
    sample = sc.transform(sample) #scale the features first with the same scaler used in training

    # Use the predict_proba method of the model to predict all class probabilities
    pred = multilr.predict_proba(sample)

    return pred[0] # since we are predicting for one sample, we just need to return the first element which is a list of 8 probabilities

# Test the function with a random sample
print(predict_label(X_test[0]))
#</Predict>
