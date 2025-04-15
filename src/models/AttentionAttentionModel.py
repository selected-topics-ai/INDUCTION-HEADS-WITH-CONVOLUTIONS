import torch
import torch.nn as nn
import torch.nn.init as init


class AttentionAttentionModel(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 # k_dim: int,
                 max_seq_length: int,
                 attention_heads_per_layer:int):

        super(AttentionAttentionModel, self).__init__()
        self.emb_dim = emb_dim

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        init.xavier_normal_(self.embedding.weight)
        # Для простоты буду использовать абсолютное позиционирование
        self.position_embedding = nn.Embedding(max_seq_length, emb_dim)
        init.xavier_normal_(self.position_embedding.weight)

        self.attention1 = nn.MultiheadAttention(emb_dim, num_heads=attention_heads_per_layer, batch_first=True)
        self.attention2 = nn.MultiheadAttention(emb_dim, num_heads=attention_heads_per_layer, batch_first=True)

        for p in self.attention1.parameters():
            if p.dim() > 1:
                init.xavier_normal_(p)

        for p in self.attention2.parameters():
            if p.dim() > 1:
                init.xavier_normal_(p)


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:

        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.float32)

        N, seq_len = input_ids.shape

        positions = torch.arange(0, seq_len).unsqueeze(0).repeat(N, 1).to(input_ids.device)
        positional_embeddings = self.position_embedding(positions)

        x = self.embedding(input_ids) + positional_embeddings

        x, _ = self.attention1(x, x, x)
        x, _ = self.attention2(x, x, x)

        logits = x @ self.embedding.weight.T

        return logits
