import torch
import torch.nn as nn

class solar_predictor_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, model_type='LSTM'):
        super().__init__()
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim = input_dim
        
        # Define the recurrent layer based on your choice
        if model_type == 'LSTM':
            self.rnn = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        else:
            self.rnn = nn.GRU(input_size= input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        
        # The Output Head: Converts the final Hidden State into 24 hours of prediction
        # self.fc = nn.Linear(hidden_dim, 10)
        # In __init__:
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(0.1), # LeakyReLU helps reach higher values than standard ReLU
            nn.Linear(128, 24)  # Your 24-hour forecast
        )

    def forward(self, x, h0, c0):
        # x: [batch, 48, 12] (input sequence)
        # Standard zero-init for 2 layers
        if h0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        if self.model_type == 'LSTM':
            # LSTM expects a tuple: (h0, c0)
            out, (hn, cn) = self.rnn(x, (h0, c0))
        else:
            # GRU expects ONLY h0
            out, hn = self.rnn(x, h0)
        
        # hn shape is [2, batch, 64]. 
        # hn[-1] is the output of the 2nd layer.
        out = self.fc(hn[-1]) 
        
        return out
    
