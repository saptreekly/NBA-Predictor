import pandas as pd
from scipy.stats import entropy

def asof_join(left, right, on, left_on, right_on):
    left = left.sort_values(by=left_on)
    right = right.sort_values(by=right_on)
    return pd.merge_asof(left, right, on=on, left_on=left_on, right_on=right_on)

def calculate_entropy(series):
    """
    Calculates the entropy of a series.
    
    Args:
        series (pd.Series): The series to calculate the entropy of.
        
    Returns:
        float: The entropy of the series.
    """
    # Drop NaNs and zeros
    series = series.dropna()
    series = series[series > 0]
    
    # If the series is empty, return 0
    if series.empty:
        return 0
        
        # Calculate the probability distribution
        series_sum = series.sum()
        if series_sum == 0:
            return 0 # If sum is zero, entropy is 0
        distribution = series / series_sum
    
        # Check for negative values in distribution (should not happen if series are non-negative)
        if (distribution < 0).any():
            # Handle this case, e.g., by raising an error or logging a warning
            # For now, let's return 0 as a safe fallback
            print("Warning: Negative values found in probability distribution. Returning 0 entropy.")
            return 0
    
        # Calculate the entropy
        return entropy(distribution, base=2)
