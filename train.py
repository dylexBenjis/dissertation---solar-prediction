import torch
import torch.nn as nn

import psutil
import matplotlib.pyplot as is_plt
import os

def train_my_model(model, loader, epochs , learning_rate=0.001):
    mse_loss = nn.MSELoss() # Standard for regression (predicting values)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train() # Put model in training mode
    print(f"Starting Training for {model.model_type}...")
    h0=None
    c0=None


    # Get current process to track only this Jupyter kernel's memory
    process = psutil.Process(os.getpid())

    ram_history = []

    # Loop through the number of epochs
    for epoch in range(epochs):
        total_loss = 0
        # For LSTM, we need to pass both h0 and c0.
        if model.model_type == 'LSTM':
            for x_batch, y_batch in loader:
                # 1. Clear previous gradients
                optimizer.zero_grad()
                
                # 2. Forward Pass: Make a 24-hour guess
                output = model(x_batch,h0,c0) # Let the model handle hidden state initialization
                
                # 3. Compute Loss: Compare guess to actual 24 hours in the strip
                loss = mse_loss(output, y_batch)
                
                # 4. Backward Pass: Calculate how to adjust weights
                loss.backward()
                
                # 5. Step: Actually update the weights
                optimizer.step()

                total_loss += loss.item()

        # For GRU, we only have h0, no c0
        else:
            for x_batch, y_batch in loader:
                # 1. Clear previous gradients
                optimizer.zero_grad()
                
                # 2. Forward Pass: Make a 24-hour guess
                output = model(x_batch,h0, c0) # Let the model handle hidden state initialization
                
                # 3. Compute Loss: Compare guess to actual 24 hours in the strip
                loss = mse_loss(output, y_batch)
                
                # 4. Backward Pass: Calculate how to adjust weights
                loss.backward()
                
                # 5. Step: Actually update the weights
                optimizer.step()

                total_loss += loss.item()
            
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch [{epoch+1}/{epochs}],  Loss: {avg_loss:.6f}")


                            

    print("Training Complete!")
