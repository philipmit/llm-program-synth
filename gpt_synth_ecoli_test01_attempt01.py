import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the digits dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# replace strings with numbers in y
unique_classes = pd.unique(y)  # find unique classes
class_dict = {value: index for index, value in enumerate(unique_classes)}  # create dictionary of classes {class : index}
y = y.replace(class_dict)  # replace classes with index
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.5, random_state=42)  # spliting data
scaler = StandardScaler()  # creating scaler object
X_train_scaled = scaler.fit_transform(X_train)  # normalizing X_train
# training logistic regression model
model = LogisticRegression(multi_class='ovr', solver='liblinear')
model.fit(X_train_scaled, y_train)
# Create a function to predict label
def predict_label(data):
    data = np.reshape(data, (1, -1))  # Reshape data so that it is a 2D array
    data_scaled = scaler.transform(data)  # Scale data using the same scaler as the training data
    return model.predict_proba(data_scaled)  # Prodict probability and return