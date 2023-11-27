#<PrevData>
print('********** Load and preview the dataset and datatypes')
import pandas as pd

dataset_name='Ecoli'
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/'+dataset_name+'/'+dataset_name+'.data', header=None)

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
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.5, random_state=42)

# Scale the features 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#</PrevData>

#<Train>
print('********** Train the model using the training data, X_train, and y_train')
# Create and train the Logistic Regression Model
model = LogisticRegression(multi_class='ovr')
model.fit(X_train, y_train)
#</Train>

#<EvalTrain>
print('********** Evaluate the model on the training dataset')
train_accuracy = model.score(X_train, y_train)
print("Training accuracy of the model: ", train_accuracy)
#</EvalTrain>

#<EvalTest>
print('********** Evaluate the model on the test dataset')
test_accuracy = model.score(X_test, y_test)
print("Test accuracy of the model: ", test_accuracy)
#</EvalTest>

#<Predict>
print('********** Define a function that can be used to make new predictions given one sample of data from X_test')

def predict_label(one_sample):
    one_sample = np.array(one_sample)
    one_sample = sc.transform(one_sample.reshape(1, -1))
    return model.predict_proba(one_sample) # Change model.predict to model.predict_proba to return probabilities

# Example usage: predict_label(X_test[0])
#</Predict>
