#<PrevData>
######## Load and preview the dataset and datatypes
# Import necessary libraries
import pandas as pd
# Read file
ecoli_df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', delim_whitespace=True, header=None)
ecoli_df.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']
# Preview dataset and datatypes
print(ecoli_df.shape)
print(ecoli_df.head())
print(ecoli_df.info())
print(ecoli_df.dtypes)
for col in ecoli_df.applymap(type).columns:
    print(col, ecoli_df.applymap(type)[col].unique())
print(ecoli_df.isnull().sum())
#</PrevData>
