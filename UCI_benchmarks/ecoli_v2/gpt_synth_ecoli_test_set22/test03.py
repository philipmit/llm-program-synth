#<PrepData>
######## Prepare the dataset for training
# Import necessary packages
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ecoli = df.copy()
ecoli.columns = ['Sequence_Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']

# Encoding the 'class' column with values from 0 to 7
le = LabelEncoder()
ecoli['class'] = le.fit_transform(ecoli['class'])

# Separating features and target
X = ecoli.drop(['Sequence_Name', 'class'], axis=1)
y = ecoli['class']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print('Shape of X_train:', X_train.shape)
print('Shape of y_train:', y_train.shape)
#</PrepData>
