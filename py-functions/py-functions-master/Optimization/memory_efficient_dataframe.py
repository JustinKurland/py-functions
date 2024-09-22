import numpy as np
import pandas as pd
from prettytable import PrettyTable


def reduce_mem_usage(df, verbose=True):
    """
    Iterate through all columns of a dataframe and modify the data type
    to reduce memory usage.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe to reduce memory usage.
    verbose : bool
        Controls the verbosity of the function. True by default.

    Returns:
    --------
    pandas.DataFrame
        Dataframe with reduced memory usage.

    """

    def _convert_boolean_to_int8(df, col):
        """
        Convert a boolean column to int8 to save memory.
        """
        if df[col].dtype != bool:
            return df
        df[col] = df[col].astype(np.int8)
        return df

    # Calculate the initial memory usage of the dataframe
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    
    # Create a table to display the results
    table = PrettyTable()
    table.field_names = ["Column name", "Original dtype", "New dtype", "Original memory (MB)", "New memory (MB)", "Percent reduction"]
    
    # Iterate over each column in the dataframe
    for col in df.columns:
        # Get the current column data type
        col_type = df[col].dtype
        
        # Get the initial memory usage of the current column
        orig_mem = df[col].memory_usage(deep=True) / 1024 ** 2
        
        # Check if the column is boolean
        if col_type == bool:
            # If the column is boolean, convert it to int8 to save memory
            df = _convert_boolean_to_int8(df, col)
            # Get the new memory usage of the current column
            new_mem = df[col].memory_usage(deep=True) / 1024 ** 2
        # Check if the column is not an object (i.e., it's a numeric column)
        elif col_type != object:
            # Get the minimum and maximum values of the current column
            c_min = df[col].min()
            c_max = df[col].max()
            # Check if the column is an integer type
            if str(col_type)[:3] == 'int':
                # Iterate over possible integer types and find the smallest type that can accommodate the column values
                for dtype in [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]:
                    if c_min > np.iinfo(dtype).min and c_max < np.iinfo(dtype).max:
                        df[col] = df[col].astype(dtype)
                        break
            # Check if the column is a float type
            elif str(col_type)[:5] == 'float':
                # Iterate over possible float types and find the smallest type that can accommodate the column values
                for dtype in [np.float16, np.float32, np.float64]:
                    if c_min > np.finfo(dtype).min and c_max < np.finfo(dtype).max:
                        df[col] = df[col].astype(dtype)
                        break
            # Get the new memory usage of the current column after converting it to a numeric type
            new_mem = df[col].memory_usage(deep=True) / 1024 ** 2
        # If the column is an object type, convert it to a categorical type to save memory
        else:
            df[col] = df[col].astype('category')
            # Get the new memory usage of the current column after converting it to
            new_mem = df[col].memory_usage(deep=True) / 1024 ** 2
        # Update the memory usage table
        percent_dec = (1 - new_mem / orig_mem) * 100
        table.add_row([col, col_type, df[col].dtype, f"{orig_mem:.4f}", f"{new_mem:.4f}", f"{percent_dec:.2f}%"])

    # Calculate the total memory usage before and after optimization
    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        # Print the memory usage table
        print(table)
        
        # Print the memory usage statistics
        mem_table = PrettyTable()
        mem_table.field_names = ["Memory usage statistics", "Value"]
        mem_table.add_row(["Memory usage of dataframe", f"{start_mem:.2f} MB"])
        mem_table.add_row(["Memory usage after optimization", f"{end_mem:.2f} MB"])
        mem_table.add_row(["Overall memory reduction", f"{(100 * (start_mem - end_mem) / start_mem):.1f}%"])
        print(mem_table)
    
    # Return the optimized dataframe
    return df