import numpy as np
import torch

def test_model(model, test_loader):
    model.eval()
    all_preds = []
    all_actuals = []
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            preds = model(x_batch, h0=None, c0=None) # Get just the predictions
            all_preds.append(preds.numpy())
            all_actuals.append(y_batch.numpy())

 
    return np.concatenate(all_preds), np.concatenate(all_actuals)