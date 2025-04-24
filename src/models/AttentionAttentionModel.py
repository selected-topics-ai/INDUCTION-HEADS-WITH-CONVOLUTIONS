import torch
import torch.nn as nn


class AttentionAttentionModel(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 max_seq_length: int,
                 attention_heads_per_layer:int):

        super(AttentionAttentionModel, self).__init__()
        self.emb_dim = emb_dim

        self.attention_heads_per_layer = attention_heads_per_layer

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        # Для простоты буду использовать абсолютное позиционирование
        self.position_embedding = nn.Embedding(max_seq_length, emb_dim)

        self.attention1 = nn.MultiheadAttention(emb_dim,
                                                num_heads=attention_heads_per_layer)
        self.attention2 = nn.MultiheadAttention(emb_dim,
                                                num_heads=attention_heads_per_layer)

        self.head = nn.Linear(emb_dim, vocab_size)


    def forward(self, input_ids: torch.Tensor, attention_mask) -> tuple[torch.Tensor, torch.Tensor]:

        N, seq_len = input_ids.shape

        positions = torch.arange(0, seq_len).unsqueeze(0).repeat(N, 1).to(input_ids.device)
        positional_embeddings = self.position_embedding(positions)

        emb_and_pos = self.embedding(input_ids) + positional_embeddings

        x = emb_and_pos.permute(1, 0, 2)

        attn_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, input_ids.device)

        attention_output, _ = self.attention1(x, x, x,
                                              attn_mask=attn_mask,
                                              key_padding_mask=attention_mask.to(attn_mask.dtype),
                                              need_weights=False)

        x = x + attention_output

        attention_output, attention_weights = self.attention2(x, x, x,
                                              attn_mask=attn_mask,
                                              key_padding_mask=attention_mask.to(attn_mask.dtype),
                                              need_weights=True,
                                              average_attn_weights=False)

        x = x + attention_output

        x = x.permute(1, 0, 2)

        logits = self.head(x)

        return logits, attention_weights

