from datasets import load_dataset

if __name__ == "__main__":

    train_dataset = load_dataset("roneneldan/TinyStories", split="train")
    validation_dataset = load_dataset("roneneldan/TinyStories", split="validation")

    train_dataset.save_to_disk("tiny-stories-train.hf")
    validation_dataset.save_to_disk("tiny-stories-validation.hf")