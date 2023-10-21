import numpy as np
from sklearn.metrics import mean_squared_error
import optuna
from src.data.preprocess_data import pipeline_build
from paths import Path

def optuna_optimize(X, y, fold_list, alg, num_cols, cat_cols):
    """
    Optuna-based hyperparameter optimization for time series models.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target variable.
        fold_list (list): List of time series cross-validation folds.
        alg: The time series model to be optimized.
        num_cols (list): List of numeric columns in the feature matrix.
        cat_cols (list): List of categorical columns in the feature matrix.

    Returns:
        tuple: A tuple containing the best hyperparameters and the corresponding best value.
    """
    print("Model : ", type(alg).__name__)

    def objective(trial,X,y,fold_list,alg,num_cols,cat_cols):
        """
       Evaluate a set of hyperparameters using Optuna for time series forecasting.

       Parameters:
       - trial (optuna.Trial): Optuna trial object for hyperparameter optimization.
       - X (pd.DataFrame): Input features.
       - y (pd.DataFrame): Target variable.
       - fold_list (list): List of folds for time series cross-validation.
       - alg: Time series regression algorithm (e.g., CatBoostRegressor, XGBRegressor, LGBMRegressor).
       - num_cols (list): List of numerical columns.
       - cat_cols (list): List of categorical columns.

       Returns:
       - float: Mean RMSE (Root Mean Squared Error) across all folds for the given hyperparameters.
       """
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.5, step=0.01),
            'n_estimators': trial.suggest_int('n_estimators', 10, 3000, step=10),
            'max_bin': trial.suggest_int('max_bin', 16, 2048, step=16),
            'subsample': trial.suggest_float('subsample', 0.1, 1, step=0.1),
            'max_depth': trial.suggest_int('max_depth', 6, 10),
        }
        if type(alg).__name__ == 'CatBoostRegressor':
            params.update({
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.1, 1, step=0.1),
                'loss_function': 'RMSE',
                'verbose': False
            })
        elif type(alg).__name__ == 'XGBRegressor':
            params.update({
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.1, 1, step=0.1),
                'num_leaves': trial.suggest_int('num_leaves', 10, 30),
                'verbosity': 0
            })
        elif type(alg).__name__ == 'LGBMRegressor':
            params.update({
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1, step=0.1),
                'num_leaves': trial.suggest_int('num_leaves', 10, 30),
                'verbosity': 0,
            })
        liste = []
        for i in range(len(fold_list)):
            train_indices = fold_list[i]['train']
            val_indices = fold_list[i]['validation']
            X_train = X.iloc[train_indices]
            y_train = y.iloc[train_indices]
            X_val = X.iloc[val_indices]
            y_val = y.iloc[val_indices]
            alg.set_params(**params)
            pipe = pipeline_build(alg, num_cols, cat_cols)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            liste.append(rmse)
        print(f'RMSE : {np.mean(liste)}')
        return np.mean(liste)

    study = optuna.create_study(direction='minimize', study_name='advanced_multiple_time_series')
    study.optimize(lambda trial: objective(trial, X, y, fold_list, alg, num_cols, cat_cols),
                   n_trials=Path.hyperparameter_trial_number)
    print(f"Best Params : {study.best_params}",
          f"Best Value : {study.best_value}")
    return study.best_params, study.best_value
