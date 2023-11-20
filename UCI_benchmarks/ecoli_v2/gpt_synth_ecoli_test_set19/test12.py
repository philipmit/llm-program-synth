#<PrevData>
######## Load and preview the dataset and datatypes
# Import necessary libraries
import pandas as pd
import io
# Read file
df = pd.read_csv(io.BytesIO(uploaded['Ecoli.csv']))
# Preview dataset and datatypes
print(df.shape)
print(df.head())
print(df.info())
print(df.dtypes)
for col in df.applymap(type).columns:
    print(col, df.applymap(type)[col].unique())
print(df.isnull().sum())
#</PrevData>
