
# Split the preprocessed data into training and testing sets and excluding the raw time columns (HR, MO, DY, YEAR).

def data_splitting(chester_data, lagos_data):

    # Split the datasets into training and testing sets 
    # 1. Find the exact row where Dec 30th starts
    target_start_idx = chester_data[
        (chester_data['YEAR'] == 2025) & 
        (chester_data['MO'] == 12) & 
        (chester_data['DY'] == 30)
    ].index[0]
    chester_test = chester_data.iloc[target_start_idx - 48 : target_start_idx + 24].copy()
    chester_train = chester_data[~((chester_data['YEAR'] == 2025) & 
                                   (chester_data['MO'] == 12) & 
                                   (chester_data['DY'] == 30))].copy()
    # 1. Find the exact row where Dec 30th starts
    target_start_idx = lagos_data[
        (lagos_data['YEAR'] == 2025) & 
        (lagos_data['MO'] == 12) & 
        (lagos_data['DY'] == 30)
    ].index[0]
    lagos_test = lagos_data.iloc[target_start_idx - 48 : target_start_idx + 24].copy()
    lagos_train = lagos_data[~((lagos_data['YEAR'] == 2025) & 
                                   (lagos_data['MO'] == 12) & 
                                   (lagos_data['DY'] == 30))].copy()
  
  
    # Select only the columns we want (excluding raw HR, MO, DY)
    columns = ['ALLSKY_SFC_SW_DWN', 'CLOUD_AMT', 'SZA', 'T2M', 'RH2M', 'PS', 'WS2M',
                'Hour_sine', 'Hour_cosine', 'DayOfYear_sine', 'DayOfYear_cosine']
    chester_train = chester_train[columns]
    chester_test = chester_test[columns]
    lagos_train = lagos_train[columns]
    lagos_test = lagos_test[columns]

    # RETURN the variables back to the notebook
    return chester_train, chester_test, lagos_train, lagos_test, target_start_idx