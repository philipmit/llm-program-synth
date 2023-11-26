#<Train>
print('********** Train the model using the training data, X_train and y_train')
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
#</Train>


#<Predict>
print('\n********** Define a function that can be used to make new predictions given one sample of data from X_test')
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
def predict_label(one_sample):
    # Standardize the one_sample to match the data model was trained on
    one_sample = sc.transform(one_sample.reshape(1, -1))
    # Return the class probabilities as a 1D array
    return model.predict_proba(one_sample)[0]
#</Predict>
