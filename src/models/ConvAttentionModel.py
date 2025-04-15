from typing import Optional

import torch
import torch.nn as nn

class ConvAttentionModel(nn.Module):

    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int):
        super(ConvAttentionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        # Задача скопировать предыдущий токен, чтобы K была равна Q предыдущего токена
        # Мини-домашка сделать так, чтобы свертка всегда предсказывала 1D
        self.causal_conv = nn.Conv1d(in_channels=emb_dim,
                                     out_channels=hidden_dim,
                                     kernel_size=3,
                                     padding=2)

        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = x.permute(0, 2, 1) # (batch_size, embed_dim, sequence_length)
        x = self.causal_conv(x)[:, :, :-2]
        x = x.permute(0, 2, 1) # (batch_size, sequence_length, embedd_dim)
        attn_output, _ = self.attention(x, x, x, attention_mask=attention_mask)
        x = attn_output.permute(1, 0, 2)
        x = self.fc(x)
        return x
