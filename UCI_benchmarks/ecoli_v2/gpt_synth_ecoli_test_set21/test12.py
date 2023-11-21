#<PrevData>
######## Load and preview the dataset and datatypes
import pandas as pd
df = pd.read_csv('./ecoli.data', header=None, delim_whitespace=True)
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
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer

df.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = df.iloc[:, 1:-1]  
y = df.iloc[:, -1] 

# The Logistic Regression model needs to train on all unique labels in order to predict all of them.
# To ensure this, stratify the split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)

# replace strings with numbers in y using LabelBinarizer
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# The shape of y_train and y_test should be (-1, 8), because there are 8 unique labels.
print(y_train.shape)
print(y_test.shape)

X_train=X_train.to_numpy()
X_test=X_test.to_numpy()

# Scale the features 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
print(X_train.shape)
print(X_train[0:5])
#</PrepData>

#<Train>
######## Train the model using the training data, X_train and y_train
# we do one-versus-rest
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

# Define a Logistic Regression model
logreg = LogisticRegression(solver='saga',max_iter=10000)

# Define the one-versus-rest classifier
ovr = OneVsRestClassifier(logreg)
ovr.fit(X_train, y_train)
#</Train>

#<Predict>
######## Define the predict_labels function that can be used to make new predictions using the trained model above given one sample from X_test
def predict_label(sample):
    # Reshape and scale the sample
    sample = np.array(sample).reshape(1, -1)
    sample = sc.transform(sample)

    # Return the predicted probabilities
    return ovr.predict_proba(sample)[0].tolist()
#</Predict>
