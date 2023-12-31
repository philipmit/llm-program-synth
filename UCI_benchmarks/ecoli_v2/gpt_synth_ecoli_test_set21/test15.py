#<PrevData>
######## Load and preview the dataset and datatypes
# Import necessary libraries
import pandas as pd
# Read file
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data', header=None, delim_whitespace=True)
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
# the name/label column is converted to numerical values
df.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = df.iloc[:, 1:-1]  
y = df.iloc[:, -1]  
# replace strings with numbers in y
y = y.replace(list(np.unique(y)), [0,1,2,3,4,5,6,7])
X=X.to_numpy()
y=y.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
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
from sklearn.linear_model import LogisticRegression

# One vs rest logisitic regression
model = LogisticRegression(multi_class='ovr', solver='liblinear')
model.fit(X_train, y_train)

# Testing model on training set
predictions = model.predict(X_train)
accuracy = np.mean(predictions == y_train)
print('Training accuracy:', accuracy)
#</Train>

#<Predict>
######## Define the predict_labels function that can be used to make new predictions using the trained model above given one sample from X_test
def predict_label(sample):
    '''
    The function 'predict_label' predicts the label of a given sample using the trained logistic regression model.
    The predicted label is returned as a list of probabilities for each label.
    '''
    # reshape the sample for prediction
    sample = np.reshape(sample, (1,-1))
    # Predicting the probabilities of each class
    predicted_probabilities = model.predict_proba(sample)
    # return the predicted probabilities
    return predicted_probabilities[0]

sample = X_test[0]
print("Predicted probabilities for sample test data: ", predict_label(sample))
#</Predict>
