class Path:
    """
    The Path class contains configurations and file paths for a time series project, specifically focused on Walmart sales data.

    Attributes:
        target (str): The target variable for the time series project.
        timestamp_column (str): The column representing timestamps in the data.
        root (str): The root directory for the project.
        train_path (str): The file path to the raw Walmart sales data.
        cleaned_train_path (str): The file path to the preprocessed and cleaned training data.
        models_path (str): The directory path to store trained models.
        fold_number (int): The number of folds for time series cross-validation.
        hyperparameter_trial_number (int): The number of trials for hyperparameter tuning.
        window (int): The size of the rolling window used in feature engineering.
        window_list (list of int): A list of window sizes for feature engineering.
        horizon (int): The forecast horizon for time series predictions.
        random_state (int): The random seed for reproducibility.
    """
    target = 'Weekly_Sales'
    timestamp_column = 'Date'
    root = 'C:/Users/MahmutYAVUZ/Desktop/Software/Python/kaggle/advanced_multiple_time_series/'
    train_path = root + '/data/raw/Walmart.csv'
    cleaned_train_path = root + '/data/preprocessed/cleaned_train.csv'
    models_path = root + "/models/"
    fold_number = 3
    hyperparameter_trial_number = 3
    window = 50
    window_list = [50,25,10]
    horizon = 4
    random_state = 42
