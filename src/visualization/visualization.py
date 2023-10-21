import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def date_column_info(data,num_cols,timestamp_column,unique_col, streamlit=False):
    """
    This function generates line charts for specified numeric columns in a DataFrame based on a timestamp column,
     providing insights into the data's temporal trends.

    Parameters:

    data (pandas.DataFrame): The DataFrame containing the timestamp column and numeric columns for visualization.
    num_cols (list of str): The list of numeric column names to visualize.
    timestamp_column (str): The name of the timestamp column for x-axis values in the line charts.
    unique_col (str): The unique column used for grouping the data and displaying the store information.
    streamlit (bool, optional): If True, the function uses Streamlit to display the charts. If False (default), the function uses Plotly.
    Returns:

    None
    """
    fig = make_subplots(rows=len(num_cols), cols=1, subplot_titles=num_cols)
    print("Store : ",unique_col)
    for i, col in enumerate(num_cols):
        line_chart = px.line(data, x=timestamp_column, y=col)
        line = line_chart.data[0]
        fig.add_trace(line, row=i + 1, col=1)
    num_rows = data.shape[1]
    fig.update_xaxes(title_text='Date', row=num_cols, col=1)
    fig.update_layout(showlegend=False, height=150*num_rows, width=1400)
    if not streamlit:
        fig.show()
    else:
        st.plotly_chart(fig, use_container_width=True)


def pred_visualize(y_val_,y_pred_,target,unique_col,m,i, streamlit=False):
    """
    This function visualizes predicted and actual values for multiple columns in a time series.

    Parameters:

    y_val_ (pandas.DataFrame): The DataFrame containing the actual values for the target columns.
    y_pred_ (pandas.DataFrame): The DataFrame containing the predicted values for the target columns.
    target (str): The name of the target variable.
    unique_col (str): The unique column used for grouping the data and displaying additional information.
    m (int): The week or time period identifier.
    i (int): The fold identifier.
    streamlit (bool, optional): If True, the function uses Streamlit to display the chart. If False (default), the function uses Plotly.
    Returns:

    None
    """
    for k in range(len(y_val_.columns.tolist())):
        fig = go.Figure()
        y_val_vis = y_val_.iloc[:,k]
        y_pred_vis = y_pred_.iloc[:,k]
        real_table_name = f'{target} and {unique_col} : {m} Week : {k+1}'
        fig.add_trace(go.Scatter(x=y_val_.index, y=y_val_vis, mode='lines',name=f'Real Day {real_table_name}',line_color='#247AFD'))
        fig.add_trace(go.Scatter(x=y_val_.index, y=y_pred_vis, mode='lines',name=f'Pred Day {real_table_name}',line_color='#ff0000'))
        fig.update_layout(title=f'Test Değerleri ve Tahminler Fold {i + 1}',
                  xaxis_title='Time',
                  yaxis_title='Horizon Time Steps')
        if not streamlit:
            fig.show()
        else:
            st.plotly_chart(fig, use_container_width=True)
