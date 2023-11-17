<PrevData>
######## Load and preview the dataset and datatypes
# Import necessary libraries
import pandas as pd
# File path for your data
file_path = '/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data'
# Read file
df = pd.read_csv(file_path, sep='\s+', header=None)
# Preview dataset and datatypes
print(df.shape)
print(df.head())
print(df.info())
print(df.dtypes)
for col in df.applymap(type).columns:
    print(col, df.applymap(type)[col].unique())
print(df.isnull().sum())
#</PrevData>