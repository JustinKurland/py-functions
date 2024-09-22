from fast_ml.utilities import display_all
from fast_ml.feature_selection import get_constant_features
import pandas as pd
import numpy as np
import pandas_flavor as pf


@pf.register_dataframe_method
def drop_constants_and_quasi_constants(df, threshold = 0.99):
    """
    Constant and quasi-constant function that searches the 
    features space and removes them.

    Args:
        df ([Pandas Dataframe]): 
            A Pandas Dataframe that has the totality of the 
            features that are in contention to be leveraged
            for actual models.
        threshold ([int]) optional): 
            Sets the variance threshold for features to select. Defaults to 0.99.

    Returns:
        [Pandas Dataframe]: 
            A Pandas Dataframe that has removed all the constants
            and quasi-constants from the feature space. 

    """

    # Identify and Set Threshold for Constant and Qausi-Constant Features
    constant_features = get_constant_features(df, threshold=threshold, dropna=False)

    # Put all the Constant and Quasi-Constant Features in a List
    constant_features_list = constant_features['Var'].to_list()

    # Drop all the Constant and Quasi-Constant Features from the Pandas Dataframe
    df.drop(columns = constant_features_list, inplace=True)

    return df
