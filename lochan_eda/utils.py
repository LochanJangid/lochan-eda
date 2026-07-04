# Helper Functions

def get_iqr_bounds(series):
    """return lower & upper bound by iqr method."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - IQR * 1.5
    upper_bound = Q3 + IQR * 1.5
    return lower_bound, upper_bound

def get_active_cols(all_cols, exclude):
    """It will Exclude specific cols and return active cols only"""
    if exclude is None: 
        exclude = []
    elif isinstance(exclude, str):
        exclude = [exclude]

    return [col for col in all_cols if col not in exclude]
    
