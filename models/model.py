import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),  
            nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim), 
        )
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x, mask=None):
        # Add causal mask
        if mask is None:
            seq_len = x.size(1)
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
            mask = mask.masked_fill(mask, float('-inf'))
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, block_size=32, n_layers=6, n_heads=4):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_enc = PositionalEncoding(emb_dim, max_len=block_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(emb_dim, n_heads) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        x = self.token_emb(x)
        x = self.pos_enc(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits

def load_gpt_from_checkpoint(
    checkpoint_path,
    vocab_size,
    emb_dim=64,
    block_size=32,
    n_layers=6,
    n_heads=4,
    device='cpu'
):
    model = GPT(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        block_size=block_size,
        n_layers=n_layers,
        n_heads=n_heads
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model
