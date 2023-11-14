from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
# Import the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Splitting predictors and the target
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# Convert class labels to numerical using LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
# Creating a pipeline of preprocessing and the model
pipeline = make_pipeline(StandardScaler(), LogisticRegression())
# Fitting the model to the training data
pipeline.fit(X_train, y_train)
# Defining a function to predict the label
def predict_label(X_sample):
    X_sample = np.array(X_sample).reshape(1, -1)
    prob_predict = pipeline.predict_proba(X_sample)
    return prob_predict[0]