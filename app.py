import streamlit as st
import datetime
import warnings
import pandas as pd
from paths import Path
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from src.models.trainer import Trainer
from src.visualization.visualization import date_column_info
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from src.data.preprocess_data import *
from src.features.feature_engineering import date_engineering
from src.models.hyperparameter_optimize import optuna_optimize

warnings.filterwarnings("ignore")
st.set_page_config(page_title="End_To_End_Advanced_Multiple_Time_Series_Regression",
                   page_icon="chart_with_upwards_trend", layout="wide")
st.markdown("<h1 style='text-align:center;'>Walmart Weekly Sales Prediction</h1>", unsafe_allow_html=True)
st.write(datetime.datetime.now(tz=None))
tabs = ["Data Analysis", "Visualization", "Train", "About"]
page = st.sidebar.radio("Tabs", tabs)

if page == "Data Analysis":
    df = pd.read_csv(Path.train_path)
    df = time_control_type(df, Path.timestamp_column)
    control = time_len_control(df, Path.timestamp_column)
    if control:
        unique_list = auto_detect(df, Path.timestamp_column)
        print(unique_list)
        st.write("Unique List",unique_list)
    df = date_sort(df,Path.timestamp_column,unique_list[0])
    variables = {
        "descriptions": {
            "Date": "The week of sales",
            "Store": "The store number",
            "Weekly_Sales": "Sales for the given store",
            "Holiday_Flag": "Whether the week is a special holiday week 1 – Holiday week 0 – Non-holiday week",
            "Temperature": "Temperature on the day of sale",
            "Fuel_Price": "Cost of fuel in the region",
            "CPI": "Prevailing consumer price index",
            "Unemployment": "Prevailing unemployment rate",
        }
    }
    profile = ProfileReport(df, title="Walmart Weekly Sales Prediction", variables=variables, dataset={
        "description": "One of the leading retail stores in the US, Walmart, would like to predict the sales and demand accurately."
                       " There are certain events and holidays which impact sales on each day. "
                       "There are sales data available for 45 stores of Walmart. "
                       "The business is facing a challenge due to unforeseen demands and runs out of stock some times, "
                       "due to the inappropriate machine learning algorithm. "
                       "An ideal ML algorithm will predict demand accurately and ingest factors like economic conditions including CPI, Unemployment Index, etc.",
        "url": "https://www.kaggle.com/datasets/yasserh/walmart-dataset"})
    st.title("Data Overview")
    st.write(df)
    st_profile_report(profile)

elif page == "Train":
    option = st.radio(
        'What model would you like to use for training?',
        ('XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor'))
    if option == 'XGBRegressor':
        model = XGBRegressor(random_state=Path.random_state)
    elif option == 'LGBMRegressor':
        model = LGBMRegressor(random_state=Path.random_state)
    elif option == 'CatBoostRegressor':
        model = CatBoostRegressor(random_seed=Path.random_state)
    df = pd.read_csv(Path.train_path)
    df = time_control_type(df, Path.timestamp_column)
    control = time_len_control(df, Path.timestamp_column)
    if control:
        unique_list = auto_detect(df, Path.timestamp_column)
        print(unique_list)
        st.write("Unique List", unique_list)
    df = date_sort(df, Path.timestamp_column, unique_list[0])
    df = date_engineering(df, Path.timestamp_column)
    time_type, frequency = frequency_detect(df, Path.timestamp_column)
    isStationary_adf = ADF_Test(df, Path.target, Path.timestamp_column)
    isStationary_kpss = KPSS_Test(df, Path.target, Path.timestamp_column, trend=315)
    df = editing_index(df, Path.timestamp_column, unique_list[0])
    num_cols = df.select_dtypes(include=['float', 'int']).columns.tolist()
    cat_cols = df.select_dtypes(exclude=['float', 'int']).columns.tolist()
    lagged_data = app_lag_data(df, Path.window, num_cols, unique_list[0], Path.timestamp_column)
    derived_data = app_derived_data(df, num_cols, Path.window, Path.window_list, time_type, frequency, unique_list[0],Path.timestamp_column)
    derived_data = derived_data.reset_index()
    derived_data.rename(columns={'level_0': Path.timestamp_column, 'level_1': unique_list[0]}, inplace=True)
    derived_data = editing_index(derived_data, Path.timestamp_column, unique_list[0])
    df = split_data(df,Path.window,len(df.reset_index()[unique_list[0]].unique()))
    if not isStationary_kpss:
        diff_data = app_diff_data(df, Path.window, lagged_data, derived_data, Path.target, time_type)
    final_data = merge_data(df,lagged_data,derived_data)
    if not isStationary_adf:
        target_list = [x for x in final_data.columns.tolist() if x.startswith(Path.target) and x != Path.target]
        final_data = trend_removal_log(final_data, target_list)
    X, y = split(final_data, Path.target, Path.horizon, len(df.reset_index()[unique_list[0]].unique()))
    num_cols = X.select_dtypes(include=['float', 'int']).columns.tolist()
    X_train, X_test, y_train, y_test = make_train_test_splits(X, y, 0.20,
                                                              len(df.reset_index()[unique_list[0]].unique()))
    fold_list = get_fold(X_train,Path.fold_number,len(df.reset_index()[unique_list[0]].unique()))
    forecast_distance = time_type_detect(time_type)
    best_params, best_value = optuna_optimize(X, y, fold_list, model, num_cols, cat_cols)
    model.set_params(**best_params)
    trainer = Trainer(X, y, fold_list, Path.horizon, num_cols, cat_cols, model, Path.timestamp_column, unique_list[0],
                      Path.target, Path.models_path)
    with st.spinner("Training is in progress, please wait..."):
        trainer.train_and_visualization()

elif page == "Visualization":
    df = pd.read_csv(Path.train_path)
    with st.spinner("Visuals are being generated, please wait..."):
        df = time_control_type(df, Path.timestamp_column)
        control = time_len_control(df, Path.timestamp_column)
        if control:
            unique_list = auto_detect(df, Path.timestamp_column)
            print(unique_list)
            st.write("Unique List", unique_list)
        df = date_sort(df, Path.timestamp_column, unique_list[0])
        for i in df[unique_list[0]].unique():
            st.write("Store : ",i)
            data = df[df[unique_list[0]] == i]
            num_cols = data.select_dtypes(include=['float', 'int']).columns.tolist()
            date_column_info(data, num_cols, Path.timestamp_column, i,streamlit=True)

elif page == "About":
    st.header("Contact Info")
    st.markdown("""**mahmutyvz324@gmail.com**""")
    st.markdown("""**[LinkedIn](https://www.linkedin.com/in/mahmut-yavuz-687742168/)**""")
    st.markdown("""**[Github](https://github.com/mahmutyvz)**""")
    st.markdown("""**[Kaggle](https://www.kaggle.com/mahmutyavuz)**""")
st.set_option('deprecation.showPyplotGlobalUse', False)
