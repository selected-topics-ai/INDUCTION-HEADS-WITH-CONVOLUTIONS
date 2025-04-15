import torch.nn as nn

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


def get_dataloader(tokenizer_name: str,
                   dataset_name: str,
                   batch_size: int = 4,
                   max_seq_length: int = 128,
                   num_proc: int = 4,
                   split: str = "train",
                   load_cached: bool = False,
                   select_rows: int = None) -> DataLoader:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt"
        )

    if not load_cached:
        dataset = load_dataset(dataset_name, split=split)
    else:
        dataset = load_from_disk(dataset_name)

    if select_rows is not None:
        dataset = dataset.select(range(select_rows))
    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=num_proc)

    def prepare_sequences(tokenized_data):
        return {
            "input_ids": tokenized_data["input_ids"],
            "attention_mask": tokenized_data["attention_mask"],
        }

    prepared_dataset = tokenized_dataset.map(prepare_sequences, batched=True, num_proc=num_proc)
    prepared_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    shuffle_flag = True if split == "train" else False

    return DataLoader(prepared_dataset, batch_size=batch_size, shuffle=shuffle_flag)


if __name__ == "__main__":
    # train_dataloader = get_dataloader(tokenizer_name="gpt2",
    #                                   load_cached=True,
    #                                   dataset_name="/Users/ilyamikheev/Desktop/projects/selected-topics-ai/INDUCTION-HEADS-WITH-CONVOLUTIONS/src/utils/tiny-stories-train.hf",
    #                                   split="train",
    #                                   batch_size=4,
    #                                   num_proc=1,
    #                                   max_seq_length=1024,
    #                                   select_rows=1)
    #
    # row = next(iter(train_dataloader))['input_ids']
    #
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    #
    # embeds = nn.Embedding(num_embeddings=tokenizer.vocab_size, embedding_dim=128)
    #
    # print(row.shape)
    #
    # print(embeds(row).shape)
    #
    # attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
    #
    # print(attention(embeds(row), embeds(row), embeds(row)))

    ds = load_from_disk("/Users/ilyamikheev/Desktop/projects/selected-topics-ai/INDUCTION-HEADS-WITH-CONVOLUTIONS/src/utils/tiny-stories-train.hf")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def count(examples):
        tokens = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        return {"token_count": len(tokens["input_ids"])}

    ds = ds.map(count, batched=False, num_proc=10)

    total_tokens = sum(ds["token_count"])
    print(f"Общее количество токенов: {total_tokens}")
