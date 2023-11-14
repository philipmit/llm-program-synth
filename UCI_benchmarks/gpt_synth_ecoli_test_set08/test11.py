import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
# Load the ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# Encode the classes with LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
# Ensure the data is np.array format
X = X.values
y = y.ravel()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Build a pipeline to standardize features and then apply logistic regression
pipeline = make_pipeline(StandardScaler(), LogisticRegression(multi_class='ovr', class_weight='balanced', solver='liblinear'))
# Fit the model using training data
pipeline.fit(X_train, y_train)
def predict_label(raw_data):
    """
    This function takes raw unprocessed data for a single sample and returns predicted probabilities for that sample.
    """
    raw_data = np.array(raw_data).reshape(1, -1)
    # predict probabilities
    predicted_probabilities = pipeline.predict_proba(raw_data)
    return predicted_probabilities[0]