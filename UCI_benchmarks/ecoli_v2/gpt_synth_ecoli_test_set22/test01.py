#<PrevData>
# Import necessary libraries
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Load ecoli dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data', delim_whitespace=True, 
                 header=None, names=['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class'])

# Drop 'Sequence Name' as it is not a feature for model training
df = df.drop(columns = ['Sequence Name'])

# Encode the categorical labels in 'classification' to numerical values
le = preprocessing.LabelEncoder()
df['class'] = le.fit_transform(df['class'])

# Prepare X (features) and y (target)
X = df.drop(columns = ['class'])
y = df['class']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Preview dataset and datatypes
print(X.head())
print(y.head())
#</PrevData>
