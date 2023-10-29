from transformers import AutoImageProcessor, Dinov2Model
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
    parser.add_argument("--save_dir", type=str, help="Specify the save directory", default="temp")
    
    args = parser.parse_args()
    
    if args.size is not None:
        print(f"Model selected: {args.size}")
    else:
        print("No model specified. Please provide a value for the --model flag.")
    
    image_processor = AutoImageProcessor.from_pretrained(f"facebook/dinov2-{args.size}")
    train_dataset = get_dataloader(args.batch_size, image_processor, split="train")

    #model = Dinov2Model.from_pretrained(f"facebook/dinov2-{args.size}")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

    model.eval()
    model.to("cuda")
    i = 0
    for batch in tqdm(train_dataset):
        inputs = batch["image"][0].to("cuda")
        labels = batch["label"].to("cuda")

        outputs = model(inputs)

        # save the features to a file
        # save the labels to a file
        torch.save(outputs, args.save_dir + '/features_' + str(i) + '.pt')
        torch.save(labels, args.save_dir + '/labels_' + str(i) + '.pt')
        i += 1

if __name__ == "__main__":
    main()