from typing import Optional

import torch
import torch.nn as nn
import torch.nn.init as init


class ConvAttentionModel(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 max_seq_length: int,
                 attention_heads_per_layer:int,
                 causal_kernel_size: int = 3):

        super(ConvAttentionModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        init.xavier_normal_(self.embedding.weight)

        self.position_embedding = nn.Embedding(max_seq_length, emb_dim)
        init.xavier_normal_(self.position_embedding.weight)

        # Задача скопировать предыдущий токен, чтобы K была равна Q предыдущего токена
        # Мини-домашка сделать так, чтобы свертка всегда предсказывала 1D
        self.causal_conv = nn.Conv1d(in_channels=emb_dim,
                                     out_channels=emb_dim,
                                     kernel_size=causal_kernel_size,
                                     padding=causal_kernel_size // 2
        )

        self.attention = nn.MultiheadAttention(emb_dim, num_heads=attention_heads_per_layer, batch_first=True)

        for p in self.attention.parameters():
            if p.dim() > 1:
                init.xavier_normal_(p)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        N, seq_len = input_ids.shape

        positions = torch.arange(0, seq_len).unsqueeze(0).repeat(N, 1).to(input_ids.device)
        positional_embeddings = self.position_embedding(positions)

        x = self.embedding(input_ids) + positional_embeddings

        x = x.permute(0, 2, 1)
        x = self.causal_conv(x)
        x = x.permute(0, 2, 1)

        x, _ = self.attention(x, x, x)

        logits = x @ self.embedding.weight.T

        return logits

