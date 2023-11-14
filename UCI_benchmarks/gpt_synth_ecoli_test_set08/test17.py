import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Load the Ecoli dataset
ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Separating the features and target variable
X = ecoli.iloc[:, 1:-1]  # All rows, all columns except the last one
y = ecoli.iloc[:, -1]   # All rows, only the last column
# Apply label encoding to class labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y.astype(str))
# No need to convert DataFrame to numpy array. It will automatically be converted in the train_test_split function.
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Initialize logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
# Training the model
model.fit(X_train, y_train)
# Initialize standard scaler
scaler = StandardScaler().fit(X_train)
def predict_label(raw_sample):
    # Preprocess the input
    sample = np.array(raw_sample).reshape(1, -1)
    sample = scaler.transform(sample)
    # Use the trained logistic regression model to predict the probabilities
    predicted_probabilities = model.predict_proba(sample)
    return predicted_probabilities[0]