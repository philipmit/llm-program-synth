#<PrevData>
######## Load and preview the dataset and datatypes
# Import necessary libraries
import pandas as pd
# Make sure the file path is correct
try:
    # Read file
    df = pd.read_csv('./datasets/ecoli.data', delimiter="\s+", header=None)  # Let's assume the correct file path is './datasets/ecoli.data'.
except FileNotFoundError as fnf_error:
    print(f"No such file or directory: '{fnf_error.filename}'. Please ensure the filepath is correct.")
    df = pd.DataFrame()  # return an empty dataframe

if not df.empty:
    # Preview dataset and datatypes
    print(df.shape)
    print(df.head())
    print(df.info())
    print(df.dtypes)
    for col in df.applymap(type).columns:
        print(col, df.applymap(type)[col].unique())
    print(df.isnull().sum())
#</PrevData>
