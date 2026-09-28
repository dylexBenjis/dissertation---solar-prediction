'''
*Version: 2.0 Published: 2021/03/09* Source: [NASA POWER](https://power.larc.nasa.gov/)
POWER API Multi-Point Download
This is an overview of the process to request data from multiple data points from the POWER API.
'''
import requests
import pandas as pd
import io

def fetch_datasets():
    locations = [(53.19, -2.9, 'chester'), (6.49, 3.39, 'lagos')]
    start_date= "20230101"
    end_date= "20251230"
    
    '''
    ##Construct the API request URL with the specified parameters
        # The parameters include:
            # CLRSKY_SFC_SW_DWN: Clear sky surface shortwave downward radiation
            # ALLSKY_SFC_SW_DWN: All sky surface shortwave downward radiation
            # CLOUD_AMT: Cloud amount
            # SZA: Solar zenith angle
            # PS: Surface pressure
            # RH2M: Relative humidity at 2 meters
            # T2M: Temperature at 2 meters
            # WS2M: Wind speed at 2 meters  
    ##No header and no metadata rows (header=false), so we can directly get and use the CSV with just the data rows
    '''    
    base_url = r"https://power.larc.nasa.gov/api/temporal/hourly/point?start={start_date}&end={end_date}&latitude={latitude}&longitude={longitude}&community=re&parameters=CLRSKY_SFC_SW_DWN%2CALLSKY_SFC_SW_DWN%2CCLOUD_AMT%2CSZA%2CPS%2CRH2M%2CT2M%2CWS2M&format=csv&header=false"   

    # Loop through each location and fetch the data
    for latitude, longitude, label in locations:
            api_request_url = base_url.format(start_date=start_date, end_date=end_date, longitude=longitude, latitude=latitude)
            response = requests.get(api_request_url, verify=True, timeout=30.00)

            if response.status_code == 200:
                # Use io.StringIO to treat the text response like a file
                raw_data = response.text
                csv_file = pd.read_csv(io.StringIO(raw_data))  
                # Save the clean version
                csv_file.to_csv(f"data_{label}.csv", index=False)
                print("CSV created successfully with shape:", csv_file.shape)
            else:
                print("Failed to fetch data:", response.status_code)
