#<PrevData>
import pandas as pd
# Read file
# Please note that you need to have the file 'ecoli.data' in your working directory or provide the full path to the file.
try:
    df = pd.read_csv('ecoli.data', header=None, delimiter = "\s+")
    # Preview dataset and datatypes
    print(df.shape)
    print(df.head())
    print(df.info())
    print(df.dtypes)
    for col in df.applymap(type).columns:
        print(col, df.applymap(type)[col].unique())
    print(df.isnull().sum())
except FileNotFoundError:
    print("File 'ecoli.data' does not exist in the current directory. Please make sure the file is present in the current working directory or provide the full path to the file.")
except Exception as e:
    print("An error occured: ", e)
#</PrevData>
