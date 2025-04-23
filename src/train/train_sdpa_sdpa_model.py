import torch
from transformers import AutoTokenizer
from src.train.train import train_and_evaluate
from src.train.dataloader import get_dataloader
from transformers import get_cosine_schedule_with_warmup
from src.models.AttentionAttentionModel import AttentionAttentionModel


if __name__ == "__main__":

    datasets_path = "/Users/ilyamikheev/Desktop/projects/selected-topics-ai/INDUCTION-HEADS-WITH-CONVOLUTIONS/src/utils/"
    train_dataset_path = f"{datasets_path}tiny-stories-train.hf"
    validation_dataset_path = f"{datasets_path}tiny-stories-validation.hf"

    tokenizer_name = "gpt2"
    max_seq_length = 512
    embedding_dim = 768
    attention_heads_per_layer = 12
    num_proc = 5
    train_batch_size = 8
    val_batch_size = 4
    train_select_rows = 60_000
    val_select_rows = 1_000
    epochs = 10

    train_dataloader = get_dataloader(tokenizer_name=tokenizer_name,
                                      load_cached=True,
                                      dataset_name=train_dataset_path,
                                      max_seq_length=max_seq_length,
                                      batch_size=train_batch_size,
                                      num_proc=num_proc,
                                      select_rows=train_select_rows, )

    val_dataloader = get_dataloader(tokenizer_name=tokenizer_name,
                                    load_cached=True,
                                    dataset_name=validation_dataset_path,
                                    max_seq_length=max_seq_length,
                                    batch_size=val_batch_size,
                                    num_proc=num_proc,
                                    select_rows=val_select_rows, )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    model = AttentionAttentionModel(
        vocab_size=tokenizer.vocab_size,
        emb_dim=embedding_dim,
        # k_dim=k_dim,
        attention_heads_per_layer=attention_heads_per_layer,
        max_seq_length=max_seq_length,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=epochs * len(train_dataloader),
    )

    device = "cpu"
    if torch.cuda.is_available():
        print("Using Cuda GPU")
        device = "cuda:0"
    elif torch.mps.is_available():
        print("Using Apple Silicon GPU")
        device = "mps"

    train_and_evaluate(
        epochs=epochs,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=torch.nn.CrossEntropyLoss(),
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        device=device,
        use_wandb=True,
        project_name="INDUCTION-HEADS-WITH-CONVOLUTIONS",
        run_name="Attention-Attention-1"
    )
