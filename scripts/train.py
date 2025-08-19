from data.data_loader   import get_batch
from models.model       import *
from scripts.evaluate   import generate_text
from util.metrics       import *
from util.visualization import *

import torch

def train_model(model, X_train, Y_train, X_val, Y_val, 
                optimizer, scheduler, tokenizer, criterion, block_size,
                max_iter=5000, eval_interval=100, batch_size=16, device='cpu'):
    
    train_losses, val_losses = [], []
    
    model.train()
    for step in range(1, max_iter + 1):
        xb, yb = get_batch(X_train, Y_train, batch_size, device)
        logits = model(xb)
        B, T, V = logits.shape
        loss = criterion(logits.view(B*T, V), yb.view(B*T))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        
        if step % eval_interval == 0:
            # Compute validation loss
            model.eval()
            with torch.no_grad():
                val_xb, val_yb = get_batch(X_val, Y_val, batch_size, device)
                val_logits = model(val_xb)
                val_loss = criterion(val_logits.view(B*T, V), val_yb.view(B*T))
            model.train()
            print(f"Step {step}/{max_iter} | Train Loss = {loss.item():.4f} | Val Loss = {val_loss.item():.4f}")
            print("Sample:\n" + generate_text(model, tokenizer, block_size, max_length=100, device=device, temperature=0.7, top_k=50))
            print("-" * 50)
            
            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            torch.save(model.state_dict(), f"models/saved_models/model_step_{step}.pt")

    return train_losses, val_losses
