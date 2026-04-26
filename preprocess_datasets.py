'''
#IN THIS CODE, WE PREPROCESS THE DATASETS TO MAKE THEM READY FOR TRAINING AND EVALUATION.
#THIS INCLUDE FILTERING NIGHT TIME, CONVERTING HOUR TIME TO COORDINATES (X, Y), AND SCALING/NORMALIZING THE DATA VALUES BETWEEN 0 AND 1.
'''
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



def preprocess(chester_data, lagos_data):

    # Convert hour time to coordinates (X, Y) using sine and cosine transformations
    # This allows the model to know exactly where in the day the data point is.
    chester_data['Hour_sine'] = np.sin(2 * np.pi * chester_data['HR'] / 24)
    chester_data['Hour_cosine'] = np.cos(2 * np.pi * chester_data['HR'] / 24)
    lagos_data['Hour_sine'] = np.sin(2 * np.pi * lagos_data['HR'] / 24)
    lagos_data['Hour_cosine'] = np.cos(2 * np.pi * lagos_data['HR'] / 24)

    # Convert year, month and day time to coordinates (X, Y) using sine and cosine transformations
    # This allows the model to know exactly where in the year the data point is.
    # We will use the day of the year (1-365) for this transformation.
    # Using the python function dayofyear, we can get the exact day of the year using the year,month and day columns.
    # rename function is used to rename the columns to year, month and day for the to_datetime function to work.
    chester_data['DayOfYear'] = pd.to_datetime(chester_data[['YEAR', 'MO', 'DY']].rename(columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day'})).dt.dayofyear
    lagos_data['DayOfYear'] = pd.to_datetime(lagos_data[['YEAR', 'MO', 'DY']].rename(columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day'})).dt.dayofyear
    # Now we can apply the sine and cosine transformations to the day of the year column.
    # We use 365.25 instead of 365 to account for leap years, which occur every 4 years.
    chester_data['DayOfYear_sine'] = np.sin(2 * np.pi * chester_data['DayOfYear'] / 365.25)
    chester_data['DayOfYear_cosine'] = np.cos(2 * np.pi * chester_data['DayOfYear'] / 365.25)
    lagos_data['DayOfYear_sine'] = np.sin(2 * np.pi * lagos_data['DayOfYear'] / 365.25)
    lagos_data['DayOfYear_cosine'] = np.cos(2 * np.pi * lagos_data['DayOfYear'] / 365.25)


    # Filter out night time data (where SZA > 90)
    # This is important because solar radiation values will be zero at night,
    # including them could affect the model's learning process.
    # chester_data = chester_data[chester_data['SZA'] < 90]
    # lagos_data = lagos_data[lagos_data['SZA'] < 90]


    # Scale/Normalize the data values between 0 and 1
    scaler = MinMaxScaler()
    data_columns_to_scale = ['CLRSKY_SFC_SW_DWN', 'ALLSKY_SFC_SW_DWN', 'CLOUD_AMT', 'PS', 'RH2M', 'T2M', 'WS2M', 'SZA']
    
    chester_data[data_columns_to_scale] = scaler.fit_transform(chester_data[data_columns_to_scale])
    lagos_data[data_columns_to_scale] = scaler.fit_transform(lagos_data[data_columns_to_scale])

    # Save the preprocessed datasets
    chester_data.to_csv("preprocessed_chester.csv", index=False)
    lagos_data.to_csv("preprocessed_lagos.csv", index=False)