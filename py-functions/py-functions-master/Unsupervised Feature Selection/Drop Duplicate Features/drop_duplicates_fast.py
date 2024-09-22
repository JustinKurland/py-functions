import pandas as pd
import numpy as np
import pandas_flavor as pf


@pf.register_dataframe_method
def drop_duplicate_columns(df):
    """
    Remove duplicated columns.
 
    Args:
         df ([pandas.DataFrame]):
            Dataframe of all columns to be evaluated.
    Returns:
        df ([pandas.DataFrame]): Dataframe with duplicated columns dropped. 
        Note that the first evaluated column of a duplicate is kept, all others
        are removed.

    """
    print(f'Dropping duplicate columns:')

    # Extract the Numeric Featres Regardless of Dtype
    nums = df._get_numeric_data().columns.tolist()

    # Convert All Numerics in the DataFrame to the Same Dtype
    df[nums] = df[nums].astype(float)

    # Create and Empty List for Duplicate Columns
    dupe_cols = []

    # Iterate to Find Duplicate Columns Among All Features
    for i in range(len(df.columns)):
        col_1 = df.columns[i]
        if col_1 not in dupe_cols:
            for col_2 in df.columns[i + 1:]:
                if df[col_1].equals(df[col_2]):
                    dupe_cols.append(col_2)
                    
    # Print the Duplicated Features to Be Dropped
                    print(f'\t{col_2}')
 
    # Create Final DataFrame Without Duplicates
    df = df.drop(dupe_cols, axis=1)
    
    return df
