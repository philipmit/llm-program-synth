#<PrevData>
######## Load and preview the dataset and datatypes
import pandas as pd
df = pd.read_csv('/path/to/ecoli/data', header=None) # Add the correct path
print(df.shape)
print(df.head())
print(df.info())
print(df.dtypes)
for col in df.applymap(type).columns:
    print(col, df.applymap(type)[col].unique())
print(df.isnull().sum())
#