from metaflow import Flow
import pandas as pd
import os
import matplotlib.pyplot as plt

import io
import base64
from sklearn.model_selection import train_test_split

def slice_date(df, start_date, end_date):
    df['ds'] = pd.to_datetime(df['ds'])
    filtered_df = df[(df['ds'] >= start_date) & (df['ds'] <= end_date)]
    return filtered_df

def get_data(data_path):
    """_summary_
    """
    return pd.read_excel(data_path, header=1)

def standardize_column_names(df):
    
    preferred_columns = [
    'Data Note', 'Well Name', 'Date', 'Time', 'Choke', 'FTHP', 'FTHT', 'FLP', 'Tsep', 'Psep',
    'Pmani', 'Meter Totalizer(Bbls)', 'Meter Factor', 'LiqRate', '%Water', '%Sediment',
    'BS&W', 'OilRate', 'DOF Plate size(inch)', 'GasDP(InchH20)', 'GasRate', 'GOR',
    'Sand(pptb)', 'Oil gravity (API)'
    ]
    
    existing_columns = df.columns.tolist()

    column_mapping = {}
    for i, existing_column in enumerate(existing_columns):
        if existing_column in preferred_columns:
            column_mapping[existing_column] = preferred_columns[i]
        else:
            column_mapping[existing_column] = None

    df.rename(columns=column_mapping, inplace=True)
    df = df[preferred_columns]

    return df

def upload_on_colab():
    uploaded = files.upload()
    for file_name, file_content in uploaded.items():
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_content), header=1)
        elif file_name.endswith('.xls') or file_name.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(file_content), header=1)

        # Process the DataFrame
        df = standardize_column_names(df)

        # Display the DataFrame
        display(df)


def data_processing(welltest_df):
    """_summary_
    """
    welltest_df = standardize_column_names(welltest_df)
    # Assuming you have your features in X and target values in y
    
    output_cols = ['OilRate', '%Water', 'LiqRate']
    welltest_df = standardize_column_names(welltest_df)
    welltest_drop_nan = welltest_df.dropna(axis=1).dropna(axis=0)
    y = welltest_drop_nan[output_cols]
    X_temp = welltest_drop_nan.drop(['Well Name', 'Date', 'Time', '%Water', 'OilRate'], axis=1)
    X_temp = X_temp.apply(pd.to_numeric, errors='coerce')
    dropped_indices = X_temp[X_temp.isna().any(axis=1)].index
    X_temp = X_temp.dropna(axis=0)
    y = y.drop(dropped_indices)
    date = welltest_df['Date'].drop(dropped_indices).to_frame(name='ds')
    X = pd.concat([date, X_temp], axis=1)
    y_oil = y['OilRate'].to_frame(name='y')
    y_water = y['%Water'].to_frame(name='y')
    y_liquidrate = y['LiqRate'].to_frame(name='y')
        
    selected_columns = X.select_dtypes(include=['object']).columns
    if len(selected_columns) > 0:
        X_selected = X.astype(float)
        X_dropped = X.drop(selected_columns, axis=1)
        X = pd.concat([X_dropped, X_selected], axis=1)
    
    assert X.isna().sum().sum() == 0
    return X, y_oil, y_water, y_liquidrate

def data_split(X, y1, y2, split_ratio=0.1):
    
    X_train, X_test, ytrain1, ytest1,\
    ytrain2, ytest2 = train_test_split(X, y1, y2, 
                                           test_size=split_ratio, shuffle=False, random_state=42)
    
    return X_train, X_test, ytrain1, ytest1, ytrain2, ytest2






def transform_standardscaler(self):
    """_summary_
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    self.X_train_transformed = scaler.fit_transform(self.X_train)
    self.X_test_transformed = scaler.transform(self.X_test)
    self.next(self.end)





def test_fit_plot(test_labels, test_predictions):
    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    lims = [0, 2000]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    


def plot_compare_prediction(test_predictions, test_labels):
    import matplotlib.pyplot as plt

    plt.scatter(range(len(test_predictions)), test_predictions, c='blue', marker='o', label='Actual')
    plt.scatter(range(len(test_labels)), test_labels, c='red', marker='*', label='Predicted')

    # Adding labels and legend
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()

    # Display the plot
    plt.show()

def error_hist(test_predictions, test_labels):
    error = test_labels - test_predictions
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error [MPG]')
    _ = plt.ylabel('Count')