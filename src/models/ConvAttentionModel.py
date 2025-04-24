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
        self.position_embedding = nn.Embedding(max_seq_length, emb_dim)

        # Задача скопировать предыдущий токен, чтобы K была равна Q предыдущего токена
        # Мини-домашка сделать так, чтобы свертка всегда предсказывала 1D
        self.pad = nn.ConstantPad1d(padding=(causal_kernel_size - 1, 0), value=0)
        self.causal_conv = nn.Conv1d(emb_dim, emb_dim, causal_kernel_size, padding=0)

        self.attention = nn.MultiheadAttention(emb_dim, num_heads=attention_heads_per_layer)

        self.head = nn.Linear(emb_dim, vocab_size)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:

        N, seq_len = input_ids.shape

        positions = torch.arange(0, seq_len).unsqueeze(0).repeat(N, 1).to(input_ids.device)
        positional_embeddings = self.position_embedding(positions)

        x = self.embedding(input_ids) + positional_embeddings

        x = x.permute(0, 2, 1)

        padded_x = self.pad(x)
        attention_output = self.causal_conv(padded_x)

        x = x + attention_output

        x = x.permute(0, 2, 1)
        x = x.permute(1, 0, 2)

        attn_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, input_ids.device)
        attention_output, attention_weights = self.attention(x, x, x,
                                             attn_mask=attn_mask,
                                             key_padding_mask=attention_mask.to(attn_mask.dtype),
                                             need_weights=True,
                                             average_attn_weights=False)

        x = x + attention_output
        x = x.permute(1, 0, 2)

        logits = self.head(x)

        return logits, attention_weights
