from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import numpy as np
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# standardizing the dataset and storing the mean and standard deviation
sc = StandardScaler().fit(X_train)
# standardize the training set and test set
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
# initialize Multilayer Perceptron classifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
# fitting the data to the MLP classifier
mlp.fit(X_train_std, y_train)
def predict_label(raw_data_sample):
    # Reshape raw_data_sample to 2D array
    raw_data_sample = np.reshape(raw_data_sample, (1, -1))
    # standardize the raw data sample using the same mean and standard deviation as before
    raw_data_sample_std = sc.transform(raw_data_sample)
    # Making the prediction
    return mlp.predict_proba(raw_data_sample_std)