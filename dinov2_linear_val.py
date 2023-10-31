from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import argparse
from tqdm import tqdm

def get_dataloader(batch_size, image_processor, split="validation"):
    dataset = load_dataset("imagenet-1k", split=split, streaming=True, use_auth_token=True)
    dataset = dataset.map(lambda x: {"image": image_processor(x["image"]).pixel_values, "label": x["label"]})
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, help="Specify the model size", 
                        choices=["base", "small", "large", "giant"], default="small")
    parser.add_argument("--batch_size", type=int, help="Specify the batch size", default=2)
    
    args = parser.parse_args()
    
    if args.size is not None:
        print(f"Model selected: {args.size}")
    else:
        print("No model specified. Please provide a value for the --model flag.")
    
    image_processor = AutoImageProcessor.from_pretrained(f"facebook/dinov2-{args.size}-imagenet1k-1-layer")
    dataset = get_dataloader(args.batch_size, image_processor)

    model = AutoModelForImageClassification.from_pretrained(f"facebook/dinov2-{args.size}-imagenet1k-1-layer")

    model.eval()
    model.to("cuda")

    total = 0
    correct = 0

    for batch in tqdm(dataset):
        inputs = batch["image"][0].to("cuda")
        labels = batch["label"].to("cuda")

        outputs = model(inputs)
        predictions = torch.argmax(outputs.logits, dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    print(f"Accuracy: {correct / total}")

if __name__ == "__main__":
    main()