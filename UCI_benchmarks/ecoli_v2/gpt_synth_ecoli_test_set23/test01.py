#<PrevData>
######## Load and preview the dataset and datatypes
# Import necessary libraries
import pandas as pd
# Read file
dataset_name='Ecoli'
dataset_name=dataset_name.lower()
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/'+dataset_name+'/'+dataset_name+'.data', header=None)
# Preview dataset and datatypes
print('df.shape')
print(df.shape)
print('df.head()')
print(df.head())
print('df.info()')
print(df.info())
print('df.dtypes')
print(df.dtypes)
for col in df.applymap(type).columns:
    print('df.applymap(type)[{col}].unique()'.format(col=col))
    print(df.applymap(type)[col].unique())
print('df.isnull().sum()')
print(df.isnull().sum())
#</PrevData>
