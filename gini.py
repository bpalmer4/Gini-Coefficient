import pandas as pd
import numpy as np

def gini(x):
    """Calculate the Gini coefficient for series of observed values (x). 
    where:
    * x is the array of observed values (and all x are positive, non-zero values)
      Note: x should be a numpy array or a pandas series
    * n is the number of values observed (derived from x)
    * i is the rank of the x-values when sorted in ascending order (derived from x)
    See: https://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    """
    
    # - sanity checks
    if not isinstance(x, pd.Series) and not isinstance(x, np.ndarray):
        raise TypeError('input series must be a pandas Series or a numpy ndarray')
    if x.min() <= 0:
        raise ValueError('all input values in series must be positive and non-zero')

    # let's work with numpy arrays
    if isinstance(x, pd.Series):
        x = x.to_numpy()
        
    # sort values in ascending order
    x = np.sort(x)

    # get n
    n = len(x)
        
    # i: numpy array from 1 to n inclusive
    i = np.arange(1, n+1)
    
    # - calculate the Gini coefficient
    return ((2 * i - n - 1) * x).sum() / (n * x.sum())