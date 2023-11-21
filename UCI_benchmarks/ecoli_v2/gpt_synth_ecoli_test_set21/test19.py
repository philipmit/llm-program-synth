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
### Start your code

### End your code
#</Train>

#<Predict>
######## Define the predict_labels function that can be used to make new predictions using the trained model above given one sample from X_test
### Start your code

### End your code
#</Predict>


#<Train>
######## Train the model using the training data, X_train and y_train
# Import the relevant module
from sklearn.linear_model import LogisticRegression

# Initiate the Logistic Regression model
logreg = LogisticRegression(max_iter=1000)

# Fit the model using the training data
logreg.fit(X_train, y_train)
#</Train>

#<Predict>
######## Define the predict_labels function that can be used to make new predictions using the trained model above given one sample from X_test
def predict_label(sample_x):
    probabilities = logreg.predict_proba([sample_x]) # predicting the probabilities for all 8 classes
    return probabilities[0] # return probabilities for the sample
#</Predict></PrevData>
#<Test>
######## Sample a test instance from X_test to test the predict_label function
sample_x = X_test[0]

# Use the predict_label function to predict the label probabilities of the sample test instance
probabilities = predict_label(sample_x)

# Print the predicted label probabilities
print('Predicted label probabilities:', probabilities)
print('Sum of probabilities:', sum(probabilities))
#</Test></PrepData>
