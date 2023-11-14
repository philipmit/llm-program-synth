import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# Load the ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Prepare the feature and target variables
X = ecoli.iloc[:, 1:-1]  # All rows, all columns excluding the first and last one
y = ecoli.iloc[:, -1]  # All rows, only the last column
# Transform target labels into numbers using LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Define the logistic regression model
log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
# Fit the model with train data
log_reg.fit(X=X_train, y=y_train)
def predict_label(raw_data_sample):
    # Ensure the input data is 2D
    if len(raw_data_sample.shape) == 1:
        raw_data_sample = np.expand_dims(raw_data_sample, 0)
    # Predict the class probabilities
    proba = log_reg.predict_proba(raw_data_sample)[0]
    return proba