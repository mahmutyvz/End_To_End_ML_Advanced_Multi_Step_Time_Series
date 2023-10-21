def date_engineering(data,col):
    """
    This function performs feature engineering on a datetime column in a DataFrame, extracting various date-related features.

    Parameters:

    data (pandas.DataFrame): The DataFrame containing the datetime column.
    col (str): The name of the datetime column for feature engineering.
    Returns:

    data (pandas.DataFrame): The DataFrame with additional date-related features.
    """
    data['Day'] = data[col].dt.day.astype(str)
    data['Month'] = data[col].dt.month.astype(str)
    data['Year'] = data[col].dt.year.astype(str)
    data['DayOfWeek'] = data[col].dt.dayofweek.astype(str)
    data['DayOfYear'] = data[col].dt.dayofyear.astype(str)
    data['WeekOfYear'] = data[col].dt.weekofyear.astype(str)
    data['Quarter'] = data[col].dt.quarter.astype(str)
    return data
