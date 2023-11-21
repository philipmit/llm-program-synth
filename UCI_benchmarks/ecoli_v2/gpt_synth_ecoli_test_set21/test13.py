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
from sklearn.preprocessing import StandardScaler, LabelEncoder

ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  
y = ecoli.iloc[:, -1]  
# replace strings with numbers in y
le = LabelEncoder()
y = le.fit_transform(y)
X=X.to_numpy()
y=y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
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
### Start your code
from sklearn.linear_model import LogisticRegression

# Initialize the model
LR = LogisticRegression(random_state=0, multi_class='auto', solver='lbfgs')

# Fit the model to the data
LR.fit(X_train, y_train)
### End your code
#</Train>

#<Predict>
######## Define the predict_labels function that can be used to make new predictions using the trained model above given one sample from X_test
### Start your code
def predict_label(sample):
    # reshape sample to be 2D as model expects input to be in 2D
    sample = np.array(sample).reshape(1, -1)
    # scale the sample
    sample = sc.transform(sample)
    # predict the class label for the sample
    pred = LR.predict_proba(sample)
    return pred[0]
### End your code
#</Predict>
