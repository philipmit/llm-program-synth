#<PrevData>
print('********** Load and preview the dataset and datatypes')
import pandas as pd

dataset_name='Ecoli'
dataset_name=dataset_name.lower()
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/'+dataset_name+'/'+dataset_name+'.data', header=None)

print('*******************')
print('df.shape')
print(df.shape)
print('*******************')
print('df.head()')
print(df.head())
print('*******************')
print('df.info()')
print(df.info())
print('*******************')
print('df.dtypes')
print(df.dtypes)
print('*******************')
for col in df.applymap(type).columns:
    print('df.applymap(type)[{col}].unique()'.format(col=col))
    print(df.applymap(type)[col].unique())
print('*******************')
print('df.isnull().sum()')
print(df.isnull().sum())
#</PrevData>
#<PrepData>
print('********** Prepare the dataset for training')
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

df = pd.DataFrame(df[0].str.split().tolist())

df.columns = ['protein_id', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']

df = df.drop(columns=['protein_id'])

le = preprocessing.LabelEncoder()
df['class'] = le.fit_transform(df['class'])

df = df.applymap(float)

X = df.iloc[:, :-1]  
y = df.iloc[:, -1]   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)

print('*******************')
print('X_train.shape')
print(X_train.shape)
print('*******************')
print('y_train.shape')
print(y_train.shape)
print('*******************')
print('X_train[0:5]')
print(X_train[0:5])
print('*******************')
print('y_train[0:5]')
print(y_train[0:5])
#</PrepData>
#<Train>
print('********** Train the model using the training data, X_train and y_train')
model = LogisticRegression()
model.fit(X_train, y_train)
#</Train>
#<Predict>
print('********** Define a function that can be used to make new predictions given one sample of data from X_test')

def predict_label(one_sample):
    # Convert list to numpy array
    one_sample = np.array(one_sample)
    # Reshape, because the model expects a 2D array
    one_sample = one_sample.reshape((1, -1))
    # Standardize the one_sample to match the data model was trained on
    one_sample = sc.transform(one_sample)
    # Return the class probabilities as a 1D array
    return model.predict_proba(one_sample)[0]  
#</Predict>
