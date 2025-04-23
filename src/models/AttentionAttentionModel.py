import torch
import torch.nn as nn
import torch.nn.init as init


class AttentionAttentionModel(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 max_seq_length: int,
                 attention_heads_per_layer:int):

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

        x = self.embedding(input_ids) + positional_embeddings
        x = x.permute(1, 0, 2)

        query = x
        key = query
        value = query

        key_padding_mask = attention_mask == 0
        causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)

        expanded_key_padding_mask = key_padding_mask.unsqueeze(1).expand(-1, seq_len, -1)
        expanded_key_padding_mask = expanded_key_padding_mask.permute(1, 0, 2)

        combined_mask = causal_mask.unsqueeze(0) + expanded_key_padding_mask.float()

        # if attention_mask is not None:
        #     attention_mask = attention_mask.to(dtype=torch.float32)
        #
        # N, seq_len = input_ids.shape
        #
        # positions = torch.arange(0, seq_len).unsqueeze(0).repeat(N, 1).to(input_ids.device)
        # positional_embeddings = self.position_embedding(positions)
        #
        # x = self.embedding(input_ids) + positional_embeddings
        #
        # causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        #
        # mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1).to(input_ids.device)

        x, _ = self.attention1(query, key, value, attn_mask=combined_mask)
        x, _ = self.attention2(x, x, x, attn_mask=combined_mask)

        x = x.permute(1, 0, 2)

        logits = x @ self.embedding.weight.T

        return logits
