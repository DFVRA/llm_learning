import torch
import math

def position_embedding(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1) # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log((10000.0) / d_model))) # (d_model//2, )
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term) # position 广播为 (seq_len, d_model//2)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

if __name__ == "__main__":
    pe = position_embedding(2, 4)
    print(pe)