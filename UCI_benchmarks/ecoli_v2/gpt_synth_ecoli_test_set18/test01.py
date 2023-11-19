#<PrevData>
######## Load and preview the dataset and datatypes
# Import necessary libraries
import pandas as pd

# Read file
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', header=None, sep="\s+")

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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Remove the identifier column
df = df.iloc[:, 1:]

# Define features, X, and labels, y
X = df.iloc[:, :-1]  # All rows, all columns except the last one
y = df.iloc[:, -1]   # All rows, only the last column
y = y.replace(list(np.unique(y)), list(range(len(np.unique(y)))))

X=X.to_numpy()
y=y.to_numpy()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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
model = LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42)
model.fit(X_train, y_train)
#</Train>

#<Eval>
######## Evaluate the model using the test data, X_test and y_test
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Scale the X_test to match X_train data features before evaluation
X_test = sc.transform(X_test)
y_pred = model.predict(X_test)

print("Model Accuracy: ", accuracy_score(y_test,y_pred))
print("Classification Report: ", classification_report(y_test,y_pred))

# Converting predicted probabilities into label encoded
prediction_prob = model.predict_proba(X_test)
prediction_label = []
for pred in prediction_prob:
    prediction_label.append(list(pred).index(max(list(pred))))

auc=roc_auc_score(y_test, prediction_label, multi_class='ovr')
print('Validation AUC: ' + str(auc))
#</Eval>

#<Predict>
######## Define a corrected function that can be used to make new predictions given one or more raw samples of data
def predict_label(raw_samples, model=model, sc=sc):
    # Expect the raw_samples as a 2D array and standardize it to match the data model was trained on
    if len(raw_samples.shape) == 1:
        raw_samples = np.array([raw_samples])
    raw_samples_std = sc.transform(raw_samples) 
    proba_predictions = model.predict_proba(raw_samples_std)

    return proba_predictions
#</Predict>
