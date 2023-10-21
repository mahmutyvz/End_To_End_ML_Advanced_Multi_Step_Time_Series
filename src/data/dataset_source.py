import pandas as pd


def read_dataset(file):
    """
       Read a dataset based on the specified file format.

       Parameters
       ----------
       file : str
           The name of the file to be read (e.g., "data.csv" or "data.xlsx").

       Returns
       -------
       df : pandas.DataFrame
           The read dataset.
   """
    if file.split('.')[1] == 'csv':
        data = pd.read_csv(file)
    elif file.split('.')[1] == 'xlsx':
        data = pd.read_excel(file)
    return data
