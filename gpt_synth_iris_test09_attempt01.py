from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Define a preprocessing function
def preprocess_input(raw_input):
    # Normalize the raw input
    raw_input = standard_scaler.transform([raw_input])
    return raw_input  
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Normalization of the features
standard_scaler = StandardScaler()
X_train = standard_scaler.fit_transform(X_train)
#Create an MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train, y_train)
#Define the prediction function
def predict_label(raw_unprocessed_data):
    processed_data = preprocess_input(raw_unprocessed_data)
    predictions = mlp.predict_proba(processed_data)
    return predictions