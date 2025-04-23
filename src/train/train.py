import os
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from tqdm import tqdm
from torch.optim.lr_scheduler import LRScheduler

from datetime import datetime
from src.models.AttentionAttentionModel import AttentionAttentionModel
from src.models.ConvAttentionModel import ConvAttentionModel


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
                       checkpoint_freq=1,
                       use_wandb=False,  # Флаг для использования wandb
                       project_name="my_project",  # Название проекта в wandb
                       run_name=None) -> tuple[list[float], list[int], list[float], list[float]]:

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    model.to(device)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    if use_wandb:
        wandb.init(project=project_name, name=run_name)
        wandb.watch(model)

    for epoch in range(epochs):

        model.train()
        total_loss, correct, total = 0, 0, 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=True)

        for i, batch in enumerate(progress_bar):

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
            scheduler.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=-1)
            correct += ((predicted == labels) * attention_mask).sum().item()
            total += attention_mask.sum().item()

            avg_train_loss = total_loss / (i + 1)
            avg_train_accuracy = correct / total

            progress_bar.set_postfix({
                "Train Loss": f"{avg_train_loss:.4f}",
                "Train Acc": f"{avg_train_accuracy:.4f}"
            })

            if use_wandb:
                wandb.log({
                    "train_loss_iter": avg_train_loss,
                    "train_accuracy_iter": avg_train_accuracy,
                    "epoch": epoch + (i + 1) / len(train_loader)  # Дробное значение эпохи
                })

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

        if use_wandb:
            wandb.log({
                "train_loss_epoch": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "epoch": epoch + 1
            })

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

    if use_wandb:
        wandb.finish()

    return train_losses, val_losses, train_accuracies, val_accuracies

