import pandas as pd
from src.data.preprocess_data import pipeline_build
from src.models.metrics import metrics_calculate
from src.visualization.visualization import pred_visualize
import os
from joblib import dump

class Trainer:
    def __init__(self, X, y, fold_list, horizon, num_cols, cat_cols, alg,timestamp_column,unique_col,target,saved_model_path):
        """
        Initialize the Trainer class.

        Parameters:
        - X (pd.DataFrame): Feature data.
        - y (pd.DataFrame): Target data.
        - fold_list (list): List of dictionaries containing training and validation indices for each fold.
        - horizon (int): Number of time steps to predict into the future.
        - num_cols (list): List of numeric feature columns.
        - cat_cols (list): List of categorical feature columns.
        - alg (object): Regression algorithm object.
        - timestamp_column (str): Name of the timestamp column in the data.
        - unique_col (str): Name of the column containing unique identifiers for time series.
        - target (str): Name of the target variable.
        - saved_model_path (str): Path to save trained models.

        Returns:
        - None
        """
        self.X = X
        self.y = y
        self.fold_list = fold_list
        self.horizon = horizon
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.alg = alg
        self.saved_model_path = saved_model_path
        self.timestamp_column = timestamp_column
        self.unique_col = unique_col
        self.target = target

    def train_and_visualization(self):
        """
        Train the regression model, save it, calculate scores, and visualize predictions.

        Returns:
        - None
        """
        directory = os.path.join(self.saved_model_path)
        if not os.path.exists(os.path.join(directory, str(f'{type(self.alg).__name__}'))):
            os.makedirs(os.path.join(directory, str(f'{type(self.alg).__name__}')), exist_ok=True)
        for i in range(len(self.fold_list)):
            train_indices = self.fold_list[i]['train']
            val_indices = self.fold_list[i]['validation']
            X_train = self.X.iloc[train_indices]
            y_train = self.y.iloc[train_indices]
            X_val = self.X.iloc[val_indices]
            y_val = self.y.iloc[val_indices]
            pipe = pipeline_build(self.alg, self.num_cols, self.cat_cols)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_val)
            model_name = os.path.join(f'{directory}/{str(type(self.alg).__name__)}/{str(i)}.gz')
            dump(self.alg, model_name, compress=('gzip', 3))
            scores = metrics_calculate(y_val, y_pred, X_train)
            print(f"Fold {i + 1} Scores : {scores}")
            model_preds_columns_list = [[f'+{i + 1}_Horizon_time_step'][0] for i in range(self.horizon)]
            y_pred = pd.DataFrame(y_pred, index=X_val.index,
                                  columns=[model_preds_columns_list])
            y_pred = y_pred.sort_values(by=[self.unique_col, self.timestamp_column], ascending=[True, True])
            y_pred = y_pred.reset_index()
            y_val = y_val.reset_index()
            print(f"Train Start-End: {X_train.index[0]} - {X_train.index[-1]}")
            print(f"Validation Start-End: {X_val.index[0]} - {X_val.index[-1]}")
            indice_start = 0
            indice_ = len(y_pred) / len(y_val[self.unique_col].unique())
            indice_end = len(y_pred) / len(y_val[self.unique_col].unique())
            for m in y_val[self.unique_col].unique():
                y_val_ = y_val[y_val[self.unique_col] == m]
                y_pred_ = y_pred.iloc[int(indice_start):int(indice_)]
                y_val_ = y_val_.set_index(self.timestamp_column)
                y_val_.drop(self.unique_col, axis=1, inplace=True)
                y_pred_.drop(self.unique_col, axis=1, inplace=True)
                y_pred_.drop(self.timestamp_column, axis=1, inplace=True)
                pred_visualize(y_val_, y_pred_, self.target, self.unique_col, m, i,streamlit=True)
                indice_start += indice_end
                indice_ += indice_end
