#<PrevData>
# Load and preview the dataset and datatypes
import pandas as pd
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', header=None, delim_whitespace=True)
df.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
print(df.shape)
print(df.head())
print(df.info())
print(df.dtypes)
for col in df.applymap(type).columns:
    print(col, df.applymap(type)[col].unique())
print(df.isnull().sum())
#</PrevData>

#<PrepData>
# Prepare the dataset for training
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

X = df.iloc[:, 1:-1]  # All rows, all columns except the first and last one
y = df.iloc[:, -1]   # All rows, only the last column
le = LabelEncoder()
y = le.fit_transform(y)

X=X.to_numpy()
y=y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Scale the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
#</PrepData>

#<Train>
# Train the model using the training data
from sklearn.multiclass import OneVsRestClassifier

# Create an instance of Logistic Regression Classifier and fit the data.
lr = OneVsRestClassifier(LogisticRegression(solver='lbfgs'))
lr.fit(X_train, y_train)
#</Train>

#<Predict>
# Define a function that can be used to make new predictions given one raw sample of data
def predict_label(raw_sample):
    if raw_sample.ndim == 1:
        raw_sample = raw_sample.reshape(1, -1)
    raw_sample = sc.transform(raw_sample)
    prediction = lr.predict_proba(raw_sample)[0] # Get the first item from the predictions list
    return prediction
#</Predict>
