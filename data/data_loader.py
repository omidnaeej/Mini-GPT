import pandas as pd
import os
import torch

def load_friends_dialogue(csv_path, max_lines=None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file {csv_path} not found")
    df = pd.read_csv(csv_path)
    if 'text' not in df.columns:
        raise ValueError("CSV file must contain a 'text' column")
    df = df.dropna(subset=['text'])
    texts = df['text'].astype(str).tolist()
    if max_lines:
        texts = texts[:max_lines]
    full_text = "\n".join(texts)
    return full_text

def create_dataset(text, tokenizer, block_size):
    data = tokenizer.encode(text)
    xs, ys = [], []
    for i in range(0, len(data) - block_size):
        x = data[i:i+block_size]
        y = data[i+1:i+block_size+1]
        xs.append(x)
        ys.append(y)
    return torch.tensor(xs), torch.tensor(ys)

def get_batch(X, Y, batch_size, device):
    idx = torch.randint(0, X.size(0), (batch_size,))
    x = X[idx].to(device)
    y = Y[idx].to(device)
    return x, y

class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return ''.join([self.itos[i] for i in ids])

