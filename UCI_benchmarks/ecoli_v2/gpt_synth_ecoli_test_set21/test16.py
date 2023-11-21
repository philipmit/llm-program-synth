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
from sklearn.preprocessing import LabelEncoder
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  
y = ecoli.iloc[:, -1]  
# replace strings with numbers in y
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Scale the features 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#</PrepData>

#<Train>
######## Train the model using the training data, X_train and y_train
# Import LogisticRegression from sklearn
from sklearn.linear_model import LogisticRegression
# Initialize LogisticRegression 
lr = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
# Fit the model
lr.fit(X_train,y_train)
#</Train>

#<Predict>
######## Define the predict_label function that can be used to make new predictions using the trained model above given one sample from X_test
def predict_label(sample):
    # Reshape the sample to 2D array as the model expects it this way
    sample = np.array(sample).reshape(1, -1)
  
    # Use the fitted model to make predictions on the sample
    return lr.predict_proba(sample)[0] # Returns a 1D array instead of a 2D array
#</Predict>
