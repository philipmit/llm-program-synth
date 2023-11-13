from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# Normalizing the data using StandardScaler. This is required because the dataset has variables of different scales
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
# Train the logistic regression model
lr = LogisticRegression(solver='lbfgs', multi_class='auto')
lr.fit(X_train, y_train)
def predict_label(raw_data):
    # Note that we only use transform here, not fit_transform as we want to use the normalization factors from the train set
    processed_data = sc.transform(raw_data.reshape(1, -1))
    pred = lr.predict_proba(processed_data)[0]
    return pred