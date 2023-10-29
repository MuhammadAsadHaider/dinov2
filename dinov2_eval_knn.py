from transformers import AutoImageProcessor
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import argparse
from tqdm import tqdm
import os

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
    val_dataset = get_dataloader(args.batch_size, image_processor, split="validation")

    if args.size == "base":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    elif args.size == "small":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    elif args.size == "large":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    elif args.size == "giant":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')

    model.eval()
    model.to("cuda")

    i = 0
    running_score = torch.zeros((50000, 5)).to("cuda")
    running_labels = torch.ones((50000, 5)).to("cuda")
    for batch in tqdm(val_dataset):
        inputs = batch["image"][0].to("cuda")
        labels = batch["label"].to("cuda")

        outputs = model(inputs)
        total_files = len(os.listdir(args.save_dir))
        # load train features and labels
        for i in tqdm(range(total_files // 2)):
            train_features = torch.load(args.save_dir + '/features_' + str(i) + '.pt', map_location=torch.device('cuda'))
            train_labels = torch.load(args.save_dir + '/labels_' + str(i) + '.pt', map_location=torch.device('cuda'))

            similarity = (outputs @ train_features.T) / (torch.norm(outputs, dim=1).unsqueeze(1) 
             @ torch.norm(train_features, dim=1).unsqueeze(1).t())

            score = similarity.topk(min(5, similarity.shape[1]), dim=1)
            temp_score = torch.cat((running_score[(i*len(labels)):((i*len(labels))+len(labels)), :], 
                              score.values), dim=1)
            temp_labels = torch.cat((running_labels[(i*len(labels)):((i*len(labels))+len(labels)), :],
                                 train_labels[score.indices]), dim=1)
            temp_running_score = temp_score.topk(5, dim=1)
            running_score[(i*len(labels)):((i*len(labels))+len(labels)), :] = temp_running_score.values
            running_labels[(i*len(labels)):((i*len(labels))+len(labels)), :] = temp_labels[torch.arange(len(labels)).unsqueeze(1), temp_running_score.indices]
        i += 1

if __name__ == "__main__":
    main()