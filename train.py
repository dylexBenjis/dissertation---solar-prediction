import torch
import torch.nn as nn

def train_my_model(model, loader, epochs=50, learning_rate=0.001):
    criterion = nn.MSELoss() # Standard for regression (predicting values)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train() # Put model in training mode
    print(f"Starting Training for {model.model_type}...")

    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in loader:
            # 1. Clear previous gradients
            optimizer.zero_grad()
            
            # 2. Forward Pass: Make a 24-hour guess
            predictions = model(x_batch)
            
            # 3. Compute Loss: Compare guess to actual 24 hours in the strip
            loss = criterion(predictions, y_batch)
            
            # 4. Backward Pass: Calculate how to adjust weights
            loss.backward()
            
            # 5. Step: Actually update the weights
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

    print("Training Complete!")