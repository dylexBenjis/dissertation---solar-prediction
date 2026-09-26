import torch
from torch.utils.data import Dataset

class SolarDataset(Dataset):
    def __init__(self, data_df, window, forecast):
        # Converts the Pandas DataFrame into a PyTorch Tensor (decimal/float numbers)
        # This is the "raw material" the model will read
        self.data = torch.tensor(data_df.values).float()
        
        # 'window' is the input size (48 hours of history)
        self.window = window
        
        # 'forecast' is the output size (24 hours of prediction)
        self.forecast = forecast

    def __len__(self):
        # Calculates how many valid windows exist in the data.
        # We subtract (window + forecast) and add 1 because the last valid
        # starting index is len(data) - window - forecast.
        return len(self.data) - self.window - self.forecast + 1

    def __getitem__(self, idx):
        # 1. Grab 'X' (The Input):
        # We take a slice of 48 rows starting at 'idx'
        # Shape will be [48, 12] -> (48 hours, 12 features)
        x = self.data[idx : idx + self.window] 
        
        # 2. Grab 'y' (The Target):
        # We start exactly where 'x' ends (idx + window)
        # We take the next 24 rows, but ONLY the first column (Irradiance)
        # Shape will be [24]
        y = self.data[idx + self.window : idx + self.window + self.forecast, 0] 
        
        return x, y