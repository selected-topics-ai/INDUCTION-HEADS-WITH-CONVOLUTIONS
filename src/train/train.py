from typing import Any

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from transformers import get_cosine_schedule_with_warmup

from datetime import datetime
from src.train.dataloader import get_dataloader
from src.models.AttentionAttentionModel import AttentionAttentionModel
from src.models.ConvAttentionModel import ConvAttentionModel

from transformers import AutoTokenizer, AutoModel


def save_checkpoint(model: nn.Module,
                    optimizer: optim.Optimizer,
                    scheduler: LRScheduler,
                    epoch: int,
                    train_loss: float,
                    val_loss: float,
                    checkpoint_dir: str = "checkpoints"):

    os.makedirs(checkpoint_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }

    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_{timestamp}.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def train_and_evaluate(epochs: int,
                       model: AttentionAttentionModel | ConvAttentionModel,
                       optimizer: optim.Optimizer,
                       scheduler: LRScheduler,
                       train_loader: data.DataLoader,
                       val_loader: data.DataLoader,
                       criterion,
                       save_checkpoints=True,
                       checkpoint_dir="model_checkpoints",
                       device: str = "mps",
                       checkpoint_freq=1) -> tuple[list[float], list[int], list[float], list[float]]:

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}, Loss: {total_loss:.4f}"):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            labels = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            attention_mask = attention_mask[:, :-1].contiguous()

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss = (loss * attention_mask.view(-1)).sum() / attention_mask.sum()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=-1)
            correct += ((predicted == labels) * attention_mask).sum().item()
            total += attention_mask.sum().item()

        train_loss = total_loss / len(train_loader)
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                labels = input_ids[:, 1:].contiguous()
                input_ids = input_ids[:, :-1].contiguous()
                attention_mask = attention_mask[:, :-1].contiguous()

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                val_loss += loss.item()
                _, predicted = torch.max(outputs, dim=-1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.numel()

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Train Acc = {train_accuracy:.4f}, "
              f"Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy:.4f}")

        if save_checkpoints and (epoch + 1) % checkpoint_freq == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                checkpoint_dir=checkpoint_dir
            )

    return train_losses, val_losses, train_accuracies, val_accuracies


if __name__ == "__main__":

    tokenizer_name = "gpt2"
    max_seq_length = 1024
    embedding_dim = 768
    attention_heads_per_layer = 12
    # k_dim = embedding_dim
    num_proc = 10
    train_batch_size = 4
    val_batch_size = 4
    #train_select_rows = 4
    train_select_rows = None
    #val_select_rows = 1000
    val_select_rows = None
    epochs = 10

    train_dataloader = get_dataloader(tokenizer_name=tokenizer_name,
                                      load_cached=True,
                                      dataset_name="/Users/ilyamikheev/Desktop/projects/selected-topics-ai/INDUCTION-HEADS-WITH-CONVOLUTIONS/src/utils/tiny-stories-train.hf",
                                      max_seq_length=max_seq_length,
                                      batch_size=train_batch_size,
                                      num_proc=num_proc,
                                      select_rows=train_select_rows, )

    val_dataloader = get_dataloader(tokenizer_name=tokenizer_name,
                                    load_cached=True,
                                    dataset_name="/Users/ilyamikheev/Desktop/projects/selected-topics-ai/INDUCTION-HEADS-WITH-CONVOLUTIONS/src/utils/tiny-stories-validation.hf",
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
    )
