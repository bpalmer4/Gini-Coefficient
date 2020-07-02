import pandas as pd
import numpy as np

def gini(x: pd.Series)-> float:
    """Calculate the Gini coefficient for pandas Series x, where all x are positive"""
    
    # - sanity checks
    if not isinstance(x, pd.Series):
        raise TypeError('input series must be a pandas Series')
    if x.min() < 0:
        raise ValueError('all input values in series must be zero or positive')
    if x.sum() <= 0:
        raise ValueError('the sum of the input series must be greater than zero')

    # - preparation for calculation
    # sort values in ascending order
    x = x.sort_values(ascending=True)
    # i: numpy array from 1 to n
    i = np.arange(1, x.size+1)
    
    # - calculate the Gini coefficient
    return ((2 * i - x.size - 1) * x).sum() / (x.size * x.sum())
