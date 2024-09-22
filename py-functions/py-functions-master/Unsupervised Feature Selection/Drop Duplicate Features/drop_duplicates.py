import pandas as pd
import numpy as np
import pandas_flavor as pf


@pf.register_dataframe_method
def drop_duplicates(df):
    """
    Searches feature space for duplicates and removes them.

    Args:
        df ([Pandas Dataframe]): 
            A Pandas Dataframe that has the totality of the 
            features that are in contention to be leveraged
            for actual models.

    Returns:
        [Pandas Dataframe]: 
            A Pandas Dataframe that has removed all the duplicates
            from the feature space. 

    """

    # Drop all duplicate features and reverse transpose
    df = df.T.drop_duplicates(keep='first').T

    return df
