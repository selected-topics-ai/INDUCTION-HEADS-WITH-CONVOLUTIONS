from datasets import load_dataset

if __name__ == "__main__":

    dataset_name = "roneneldan/TinyStories"

    train_dataset = load_dataset(dataset_name, split="train")
    validation_dataset = load_dataset(dataset_name, split="validation")

    train_dataset.save_to_disk("tiny-stories-train.hf")
    validation_dataset.save_to_disk("tiny-stories-validation.hf")