from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from tqdm import tqdm

def train_and_evaluate(epochs: int,
                       model: nn.Module,
                       optimizer: optim.Optimizer,
                       train_loader: data.DataLoader,
                       val_loader: data.DataLoader,
                       criterion) -> tuple[list[float], list[int], list[float], list[float]]:

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            inputs, targets = batch["input_ids"], batch["labels"]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=-1)
            correct += (predicted == targets).sum().item()
            total += targets.numel()

        train_loss = total_loss / len(train_loader)
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch["input_ids"], batch["labels"]
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                val_loss += loss.item()
                _, predicted = torch.max(outputs, dim=-1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.numel()

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Train Acc = {train_accuracy:.4f}, "
              f"Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy:.4f}")

    return train_losses, val_losses, train_accuracies, val_accuracies


if __name__ == "__main__":
    train_and_evaluate()