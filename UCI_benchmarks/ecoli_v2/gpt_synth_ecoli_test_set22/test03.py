#<PrevData>
import pandas as pd
# Read file
df = pd.read_csv('ecoli.data', header=None, delimiter = "\s+")
# Preview dataset and datatypes
print(df.shape)
print(df.head())
print(df.info())
print(df.dtypes)
for col in df.applymap(type).columns:
    print(col, df.applymap(type)[col].unique())
print(df.isnull().sum())
#</PrevData>
