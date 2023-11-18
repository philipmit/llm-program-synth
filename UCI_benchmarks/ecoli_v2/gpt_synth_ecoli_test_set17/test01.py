#<PrevData>
######## Load and preview the dataset and datatypes
# Import necessary libraries
import pandas as pd
import numpy as np
# Read file
df = pd.read_csv('/data/sls/scratch/pschro/p2/data/UCI_benchmarks/ecoli/ecoli.data', header=None)
# Preview dataset and datatypes
print(df.shape)
print(df.head())
print(df.info())
print(df.dtypes)
for col in df.applymap(type).columns:
    print(col, df.applymap(type)[col].unique())
print(df.isnull().sum())
#</PrevData>
#<PrepData>
######## Prepare the dataset for training
# The data is not properly loaded. It seems like it is not comma separated values.
# So, we need to split the string into multiple columns and then remove the unnecessary column

df = df[0].str.split(expand=True)
df.drop(0, axis=1, inplace=True)  # Drop the unnecessary column

# Now let's get the feature matrix X and target vector y
X = df.iloc[:, :-1]  # All rows, all columns except the last one
y = df.iloc[:, -1]    # All rows, only the last column
# Replace the unique labels with unique numbers
y = y.replace(list(np.unique(y)), list(range(len(np.unique(y)))))
#</PrepData>
