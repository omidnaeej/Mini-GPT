from data.data_loader   import *
from models.model       import *
from scripts.train      import *
from scripts.evaluate   import *
from util.metrics       import *
from util.visualization import *

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

def main(config_path="config/config.yaml"):
    # load config
    with open(config_path) as f: cfg = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
        
    # download & prepare data
    text = load_friends_dialogue("data/dataset/friends.csv")
    
    tokenizer = CharTokenizer(text)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")


    # Split into train and validation sets
    X, Y = create_dataset(text, tokenizer, cfg['block_size'])
    train_size = int(0.9 * len(X))
    X_train, Y_train = X[:train_size], Y[:train_size]
    X_val, Y_val = X[train_size:], Y[train_size:]
    print(f"Training dataset shape: {X_train.shape}")
    print(f"Validation dataset shape: {X_val.shape}")

    model = GPT(
        vocab_size=tokenizer.vocab_size,
        emb_dim=cfg['n_embd'],
        block_size=cfg['block_size'],
        n_layers=cfg['n_layers'],
        n_heads=cfg['n_head']
    ).to(device)

    print_model_param_counts(model, name="MyGPT")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['learning_rate']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['max_iter'])

    train_loss, val_loss = train_model(
        model=model,
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=tokenizer,
        criterion=criterion,
        block_size=cfg['block_size'],
        max_iter=cfg['max_iter'],
        eval_interval=cfg['eval_interval'],
        batch_size=cfg['batch_size'],
        device=device
    )
    
    plot_loss(train_loss, val_loss)

    print("Sampling:")
    print(generate_text(model, tokenizer, cfg['block_size'], 
                        start_text="Joey: ", temperature=cfg['temperature'],
                        top_k=5, max_length=cfg['max_length'], 
                        device=device))
    

    print("\nBeam Search:")
    print(generate_text(model, tokenizer, cfg['block_size'], 
                        start_text="Monica: ", temperature=cfg['temperature'], 
                        beam_width=cfg['beam_width'], max_length=cfg['max_length'], 
                        device=device))

    
    print("\nHow can I help you?")
    while True:
        prompt = input("Enter your prompt: ")
        if prompt == "exit":
            break
        print(generate_text(model, tokenizer, cfg['block_size'], 
                            start_text=prompt, temperature=cfg['temperature'], 
                            beam_width=cfg['beam_width'], max_length=cfg['max_length'], 
                            device=device))


if __name__=="__main__":
    main()
