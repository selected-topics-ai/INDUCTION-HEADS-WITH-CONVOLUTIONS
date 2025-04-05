import torch
import torch.nn as nn

class ConvAttentionModel(nn.Module):

    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int):
        super(ConvAttentionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.casual_conv = nn.Conv1d(in_channels=emb_dim, out_channels=hidden_dim, kernel_size=3, padding=2)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=1)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.permute(0, 2, 1) # (batch_size, embed_dim, sequence_length)
        x = self.casual_conv(x)[:, :, :-2]
        x = x.permute(2, 0, 1)
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.permute(1, 0, 2)
        x = self.fc(x)
        return x
