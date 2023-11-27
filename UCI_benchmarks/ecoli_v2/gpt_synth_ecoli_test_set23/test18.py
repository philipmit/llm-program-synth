#<PrevData>
print('********** Load and preview the dataset and datatypes')
# Import necessary libraries
import pandas as pd
# Read file
dataset_name='Ecoli'
dataset_name=dataset_name.lower()
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/'+dataset_name+'/'+dataset_name+'.data', header=None)
# Preview dataset and datatypes
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
# Import necessary packages
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Preprocessing
df = df[0].str.split(expand=True)
df.iloc[:, 1:-1] = df.iloc[:, 1:-1].apply(pd.to_numeric)
df.columns = range(df.shape[1])

# Define features, X, and labels, y
X = df.iloc[:, 1:-1]  # exclude first and last column
y = df.iloc[:, -1]   # All rows, only the last column
y = y.replace(list(np.unique(y)), list(range(len(np.unique(y)))))

# Convert to numpy arrays for use with scikit-learn 
X=X.to_numpy()
y=y.to_numpy()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)

# Scale the features 
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
#<EvalTrain>
print('********** Evaluate the model on the training dataset')

# Measure its accuracy on the training set
train_accuracy = model.score(X_train, y_train)

print("Training accuracy of the model: ", train_accuracy)
#</EvalTrain>

#<TransformTest>
print('********** Transform the test dataset in the same way as the training dataset')
X_test = sc.transform(X_test)

print('*******************')
print('X_test.shape')
print(X_test.shape)
print('*******************')
print('X_test[0:5]')
print(X_test[0:5])
#</TransformTest>

#<EvalTest>
print('********** Evaluate the model on the test dataset')
# Measure its accuracy on the test set
test_accuracy = model.score(X_test, y_test)

print("Test accuracy of the model: ", test_accuracy)
#</EvalTest>

#<Predict>
print('********** Define a function that can be used to make new predictions given one sample of data from X_test')
def predict_label(one_sample):
    # Standardize the one_sample to match the data model was trained on
    one_sample = sc.transform(one_sample.reshape(1, -1))
    # Return the class label
    return model.predict(one_sample)[0]  
# Example usage: predict_label(X_test[0])
#</Predict>
