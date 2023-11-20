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

X = df.iloc[:, 1:-1].values  # All rows, all columns except the first and last one
y = df.iloc[:, -1].values   # All rows, only the last column

# replace strings with numbers in y
np.unique(y)
len(list(np.unique(y)))
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7])

# Perform label encoding
le = LabelEncoder()
y = le.fit_transform(y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
#</PrepData>

#<Train>
# Train the model using the training data
from sklearn.multiclass import OneVsRestClassifier

# Create an instance of Logistic Regression Classifier and fit the data.
model = OneVsRestClassifier(LogisticRegression(solver='lbfgs'))
model.fit(X_train, y_train)
#</Train>

#<Predict>
# Corrected predict_label function to handle single sample or multiple samples
def predict_label(raw_sample):
    if len(raw_sample.shape) == 1:
        raw_sample = raw_sample.reshape(1, -1)
    raw_sample = sc.transform(raw_sample)
    return model.predict_proba(raw_sample)[0]  # Corrected this line to return 1D array instead of 2D
#</Predict>
