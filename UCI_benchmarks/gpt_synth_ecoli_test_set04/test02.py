import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
X = ecoli.drop(['Sequence Name', 'class'], axis=1)  
y = ecoli['class']  
# Replace strings with numbers in y
class_labels = np.unique(y)
label_dict = {label: index for index, label in enumerate(class_labels)}
y = y.replace(label_dict)
X = X.to_numpy()
y = y.to_numpy()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.5, random_state=42)
# normalize the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Define Logistic Regression model
# The 'multi_class' parameter is set to 'multinomial' as we have multiple classes in the target label. 
# The solver is initialized as 'lbfgs' which is suitable for multiclass problems.
log_model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
log_model.fit(X_train, y_train)
def predict_label(raw_sample):
    """Predicts the label of a single sample using the trained model."""
    raw_sample = np.array(raw_sample).reshape(1, -1)
    sample = scaler.transform(raw_sample)
    probas = log_model.predict_proba(sample)
    return probas[0]