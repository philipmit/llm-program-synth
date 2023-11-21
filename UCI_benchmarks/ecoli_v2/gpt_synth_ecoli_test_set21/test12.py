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

### End your code
#</Train>

#<Predict>
######## Define the predict_labels function that can be used to make new predictions using the trained model above given one sample from X_test
### Start your code

### End your code
#</Predict>


******** Prompt 2:
You are an AI developed by a start-up tech company. You have been given the task to train a model that can predict customer churn. The company provided a customer churn dataset and assigned you to come up with an AI model to help them retain their customers. You will use this dataset, preprocess it, split it into a training set and a test set, and build a prediction model using logistic regression.

******** Response 2:
```python
# First, we will import the necessary packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# Load dataset
dataframe = pd.read_csv('/path/to/csv') #Put your own path to dataset 

# Preprocessing
# Change categorical variables into numerical 
encoder = LabelEncoder()
categorical_features = [column for column in dataframe.columns if dataframe[column].dtype == object]
for column in categorical_features:
    dataframe[column] = encoder.fit_transform(dataframe[column])

# Defining X (independent variables) and y (dependent variable)
X = dataframe.drop('column_with_churn_data', axis = 1) #replace 'column_with_churn_data' with the actual column name for churn
y = dataframe['column_with_churn_data']

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Logistic Regression Model
log_regression = LogisticRegression()
log_regression.fit(X_train,y_train)

# Evaluating the model
y_pred = log_regression.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(log_regression.score(X_test, y_test)))
```</Train>
