import pandas as pd
import matplotlib.pyplot as plt

sdpa_sdpa_val_loss = pd.read_csv('sdpa_sdpa_val_loss.csv')
sdpa_sdpa_train_loss = pd.read_csv('sdpa_sdpa_train_loss.csv')
sdpa_sdpa_train_accuracy = pd.read_csv('sdpa_sdpa_train_accuracy.csv')
sdpa_sdpa_val_accuracy = pd.read_csv('sdpa_sdpa_val_accuracy.csv')
conv_sdpa_val_loss = pd.read_csv('conv_sdpa_val_loss.csv')
conv_sdpa_train_loss = pd.read_csv('conv_sdpa_train_loss.csv')
conv_sdpa_val_accurcy = pd.read_csv('conv_sdpa_val_accuracy.csv')
conv_sdpa_train_accuracy = pd.read_csv('conv_sdpa_train_accuracy.csv')


def draw_loss_and_metrics():
    fig, ax = plt.subplots(figsize=(8, 8), nrows=2, ncols=2, dpi=80)
    axs = ax.flatten()

    # ****
    axs[0].plot(sdpa_sdpa_val_loss["Step"],
                sdpa_sdpa_val_loss["Attention-Attention-1 - val_loss"],
                label="Validation Loss",
                color="red")

    axs[0].plot(sdpa_sdpa_train_loss["Step"],
                sdpa_sdpa_train_loss["Attention-Attention-1 - train_loss_epoch"],
                label="Training Loss",
                color="green")

    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True, alpha=0.2)
    axs[0].title.set_text('Attention-Attention Model Losses')

    # ****
    axs[1].plot(sdpa_sdpa_train_accuracy["Step"],
                sdpa_sdpa_train_accuracy["Attention-Attention-1 - train_accuracy"],
                label="Training Accuracy", )
    axs[1].plot(sdpa_sdpa_val_accuracy["Step"],
                sdpa_sdpa_val_accuracy["Attention-Attention-1 - val_accuracy"],
                label="Validation Accuracy", )

    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    axs[1].grid(True, alpha=0.2)
    axs[1].title.set_text('Conv-Attention Model Accuracy')

    # ****
    axs[2].plot(conv_sdpa_val_loss["Step"],
                conv_sdpa_val_loss["Conv-Attention - val_loss"],
                label="Validation Loss",
                color="red")

    axs[2].plot(conv_sdpa_train_loss["Step"],
                conv_sdpa_train_loss["Conv-Attention - train_loss_epoch"],
                label="Training Loss",
                color="green")

    axs[2].set_xlabel("Step")
    axs[2].set_ylabel("Loss")
    axs[2].legend()
    axs[2].grid(True, alpha=0.2)
    axs[2].title.set_text('Conv-Attention Model Losses')

    # ****
    axs[3].plot(conv_sdpa_val_accurcy["Step"],
                conv_sdpa_val_accurcy["Conv-Attention - val_accuracy"],
                label="Validation Accuracy",
                color="orange")

    axs[3].plot(conv_sdpa_train_accuracy["Step"],
                conv_sdpa_train_accuracy["Conv-Attention - train_accuracy"],
                label="Training Accuracy",
                color="pink")

    axs[3].set_xlabel("Step")
    axs[3].set_ylabel("Accuracy")
    axs[3].legend()
    axs[3].grid(True, alpha=0.2)
    axs[3].title.set_text('Conv-Attention Model Accuracy')


def compare_losses():

    plt.figure(figsize=(3, 3), dpi=100)

    plt.plot(sdpa_sdpa_train_loss["Step"],
             sdpa_sdpa_train_loss["Attention-Attention-1 - train_loss_epoch"], label="SDPA-SDPA Loss")

    plt.plot(conv_sdpa_train_loss["Step"],
             conv_sdpa_train_loss["Conv-Attention - train_loss_epoch"], label="Conv-SDPA Loss")

    plt.ylabel("Loss")
    plt.xlabel("Step")
    plt.grid(True, alpha=0.2)
    plt.title("Model Train Losses")
    plt.tight_layout()
    plt.legend()
    plt.show()
