import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np
def load_data():
    ecoli = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
    ecoli.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
    y = ecoli.iloc[:, -1] 
    le = LabelEncoder()
    y = le.fit_transform(y)
    X=ecoli.iloc[:, 1:-1].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    return X_train, y_train, le
def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model
def predict_label(model, le, raw_input):
    raw_input_transformed = np.array(raw_input[1:-1]).astype(float)
    raw_input_transformed = raw_input_transformed.reshape(1, -1)  # Reshape the data to 2D array
    prediction = model.predict_proba(raw_input_transformed)
    prediction_label = le.inverse_transform([np.argmax(prediction)])
    return prediction_label, prediction
X_train, y_train, le = load_data()
model = train_model(X_train, y_train)
raw_input = ['AT1G42900.1', 0.44, 0.51, 0.5, 0.61, 0.44, 0.34, 0.38, 'cyt']
predicted_label, predicted_probabilities = predict_label(model, le, raw_input)
print("Predicted Label: ", predicted_label)
print("Predicted Probabilities: ", predicted_probabilities)