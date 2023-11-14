import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1].to_numpy()  # All rows, all columns except the last one
# create labelencoder object
le = LabelEncoder()
# convert the categorical columns into numeric
y = le.fit_transform(ecoli.iloc[:, -1]) # All rows, only the last column
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
#Training the model
model = LogisticRegression(multi_class='multinomial', max_iter=1000, solver='lbfgs')
model.fit(X_train, y_train)
# Defining the function predict_label
def predict_label(sample):
    return model.predict_proba([sample])[0]