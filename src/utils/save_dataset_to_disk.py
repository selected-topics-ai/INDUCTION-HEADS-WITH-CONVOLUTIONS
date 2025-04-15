from datasets import load_dataset, load_from_disk

if __name__ == "__main__":

    # train_dataset = load_dataset("roneneldan/TinyStories", split="train")
    validation_dataset = load_dataset("roneneldan/TinyStories", split="validation")

    # train_dataset.save_to_disk("tiny-stories-train.hf")
    # validation_dataset.save_to_disk("tiny-stories-validation.hf")

    print(len(load_from_disk('/Users/ilyamikheev/Desktop/projects/selected-topics-ai/INDUCTION-HEADS-WITH-CONVOLUTIONS/src/utils/tiny-stories-validation.hf')))
