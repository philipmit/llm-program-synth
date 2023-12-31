#<PrevData>
print('********** Load and preview the dataset and datatypes')
# Import necessary libraries
import pandas as pd

# File paths
TRAIN_DATA_FILE = "/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data"

# Read file
df = pd.read_csv(TRAIN_DATA_PATH, header=None, delim_whitespace=True)

# Preview dataset and datatypes
print('*******************')
print('df.shape')
print(df.shape)
print('*******************')
print('df.head()')
print(df.head())
print('*******************')
print('df.info()')
print(df.info())
print('*******************')
print('df.dtypes')
print(df.dtypes)
print('*******************')
for col in df.applymap(type).columns:
    print('df.applymap(type)[{col}].unique()'.format(col=col))
    print(df.applymap(type)[col].unique())
print('*******************')
print('df.isnull().sum()')
print(df.isnull().sum())
#</PrevData>
