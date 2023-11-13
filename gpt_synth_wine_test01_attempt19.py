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
# normalize the features for better prediction
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
# Apply Logistic Regression
lr = LogisticRegression(multi_class='ovr', solver='liblinear')
lr.fit(X_train, y_train)
def predict_label(raw_data):
    normalized_data = sc.transform([raw_data])
    return lr.predict_proba(normalized_data)