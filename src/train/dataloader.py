
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def get_dataloader(tokenizer_name: str,
                   dataset_name: str,
                   split: str,
                   batch_size: int) -> DataLoader:

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

    dataset = load_dataset(dataset_name, split=split)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    def prepare_sequences(tokenized_data):
        input_ids = tokenized_data["input_ids"]
        labels = tokenized_data["input_ids"].clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100
        return {"input_ids": input_ids, "labels": labels}

    prepared_dataset = tokenized_dataset.map(prepare_sequences, batched=True)
    prepared_dataset.set_format(type="torch", columns=["input_ids", "labels"])

    shuffle_flag = True if split == "train" else False

    return DataLoader(prepared_dataset, batch_size=batch_size, shuffle=shuffle_flag)