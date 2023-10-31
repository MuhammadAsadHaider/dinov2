from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import argparse
from tqdm import tqdm
import os
import json
import numpy as np
import pkgutil
from torchvision.datasets import ImageFolder
from torchvision import transforms

class RealLabelsImagenet:
    def __init__(self, filenames, real_json=None, topk=(1, 5)):
        if real_json is not None:
            with open(real_json) as real_labels:
                real_labels = json.load(real_labels)
        else:
            real_labels = json.loads(
                pkgutil.get_data(__name__, os.path.join('_info', 'imagenet_real_labels.json')).decode('utf-8'))
        real_labels = {f'ILSVRC2012_val_{i + 1:08d}.JPEG': labels for i, labels in enumerate(real_labels)}
        self.real_labels = real_labels
        self.filenames = filenames
        assert len(self.filenames) == len(self.real_labels)
        self.topk = topk
        self.is_correct = {k: [] for k in topk}
        self.sample_idx = 0

    def add_result(self, output):
        maxk = max(self.topk)
        _, pred_batch = output.topk(maxk, 1, True, True)
        pred_batch = pred_batch.cpu().numpy()
        for pred in pred_batch:
            filename = self.filenames[self.sample_idx]
            filename = os.path.basename(filename)
            if self.real_labels[filename]:
                for k in self.topk:
                    self.is_correct[k].append(
                        any([p in self.real_labels[filename] for p in pred[:k]]))
            self.sample_idx += 1

    def get_accuracy(self, k=None):
        if k is None:
            return {k: float(np.mean(self.is_correct[k])) * 100 for k in self.topk}
        else:
            return float(np.mean(self.is_correct[k])) * 100

def get_dataloader(batch_size, data_dir, transforms):
    dataset = ImageFolder(root=data_dir, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    filenames = [sample[0] for sample in dataset.samples] 
    return dataloader, filenames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, help="Specify the model size", 
                        choices=["base", "small", "large", "giant"], default="small")
    parser.add_argument("--batch_size", type=int, help="Specify the batch size", default=2)
    parser.add_argument("--data_dir", type=str, help="Specify the data directory", 
                        default="val/")
    parser.add_argument("--real_labels", type=str, help="Specify the real labels json file",
                        default="real.json")
    
    args = parser.parse_args()
    
    if args.size is not None:
        print(f"Model selected: {args.size}")
    else:
        print("No model specified. Please provide a value for the --model flag.")
    
    image_processor = AutoImageProcessor.from_pretrained(f"facebook/dinov2-{args.size}-imagenet1k-1-layer")

    img_transforms = transforms.Compose([
        image_processor,
    ])

    dataset, filenames = get_dataloader(args.batch_size, args.data_dir, img_transforms)

    model = AutoModelForImageClassification.from_pretrained(f"facebook/dinov2-{args.size}-imagenet1k-1-layer")

    model.eval()
    model.to("cuda")

    real_labels = RealLabelsImagenet(filenames, real_json=args.real_labels)

    for batch in tqdm(dataset):
        inputs, _ = batch
        inputs = inputs['pixel_values'][0].to("cuda")

        outputs = model(inputs)
        real_labels.add_result(outputs.logits)

    top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
    print(f"Top-1 Accuracy: {top1a}")
    print(f"Top-5 Accuracy: {top5a}")

if __name__ == "__main__":
    main()