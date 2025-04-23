import torch
import torch.nn as nn
import torch.nn.init as init


class AttentionAttentionModel(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 max_seq_length: int,
                 attention_heads_per_layer:int):

        nn.Transformer

        super(AttentionAttentionModel, self).__init__()
        self.emb_dim = emb_dim

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        init.xavier_normal_(self.embedding.weight)
        # Для простоты буду использовать абсолютное позиционирование
        self.position_embedding = nn.Embedding(max_seq_length, emb_dim)
        init.xavier_normal_(self.position_embedding.weight)

        self.attention1 = nn.MultiheadAttention(emb_dim, num_heads=attention_heads_per_layer)
        self.attention2 = nn.MultiheadAttention(emb_dim, num_heads=attention_heads_per_layer)

        for p in self.attention1.parameters():
            if p.dim() > 1:
                init.xavier_normal_(p)

        for p in self.attention2.parameters():
            if p.dim() > 1:
                init.xavier_normal_(p)


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:

        N, seq_len = input_ids.shape

        positions = torch.arange(0, seq_len).unsqueeze(0).repeat(N, 1).to(input_ids.device)
        positional_embeddings = self.position_embedding(positions)

        emb_and_pos = self.embedding(input_ids) + positional_embeddings

        x = emb_and_pos.permute(1, 0, 2)

        query = x
        key = query
        value = query

        attention_output, _ = self.attention1(query, key, value, is_causal=True)

        x = x + attention_output

        attention_output, _ = self.attention2(x, x, x, is_causal=True)

        x = x + attention_output

        x = x.permute(1, 0, 2)

        logits = x @ self.embedding.weight.T

        return logits
