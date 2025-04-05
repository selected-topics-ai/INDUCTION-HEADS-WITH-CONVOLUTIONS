import torch
import torch.nn as nn

class AttentionAttentionModel(nn.Module):

    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int):
        super(AttentionAttentionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.attention1 = nn.MultiheadAttention(emb_dim, num_heads=1)
        self.attention2 = nn.MultiheadAttention(emb_dim, num_heads=1)
        self.fc = nn.Linear(emb_dim, hidden_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input)
        x = x.permute(1, 0, 2) # (sequence_length, batch_size, embed_dim)
        x, _ = self.attention1(x, x, x)
        x, _ = self.attention2(x, x, x)
        x = x.permute(1, 0, 2)
        x = self.fc(x)
        return x
