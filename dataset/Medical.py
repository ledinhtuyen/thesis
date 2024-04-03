import os
import json
from pathlib import Path

import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def train_transform(input_size=384, meanstd={'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}):
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=meanstd['mean'], std=meanstd['std'])])
    return transform_train

def test_transform(input_size=384):
    transform_test = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=3)])
    return transform_test

def unnormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0):
    img = (img * std  + mean) * max_pixel_value
    return np.clip(np.round(img), 0, 255).astype(np.uint8)

def mean_and_std(train_data, prefix_path, meanstd_file):
    """
    Compute the mean and std of the dataset
    Args:
            img_dir: the directory of the images
            meanstd_file: the file to save the mean and std
    """

    # BEGIN CODE
    if os.path.isfile(meanstd_file):
        meanstd = torch.load(meanstd_file)
    else:
            mean = torch.zeros(3)
            std = torch.zeros(3)

            for img_path in train_data:
                img = read_image(os.path.join(prefix_path, img_path)) / 255.0
                mean += img.mean(dim=(1, 2))
                std += img.std(dim=(1, 2))
                
            mean /= len(train_data)
            std /= len(train_data)
            meanstd = {'mean': mean, 'std': std}
            torch.save(meanstd, meanstd_file)
    # END CODE
    return meanstd

class Medical(object):
    def __init__(self, prefix_path, annotation_file, type='pretrain'):
        self.prefix_path = prefix_path
        self.data = json.load(open(annotation_file))
        if type == 'pretrain':
            self.data_list = self.data["train"]
            self.train_data, self.test_data = train_test_split(self.data_list, test_size=0.005, random_state=42)

    def get_train_data(self):
        return self.train_data
    
    def get_test_data(self):
        return self.test_data

class PretrainMedical(Dataset):
    def __init__(self, data=None, json_file="", meanstd_file="", prefix_path=None, train=False, transform=None):
        self.data = data
        
        if not os.path.exists(json_file):
            with open(json_file, 'w') as f:
                json.dump(data, f)
            if train:
                mean_and_std(data, prefix_path, meanstd_file)
        elif os.path.exists(json_file):
            self.data = json.load(open(json_file))
            self.meanstd = torch.load(meanstd_file)
        elif not os.path.exists(json_file) and data is None:
            raise ValueError("json_file are None")

        self.transform = transform
        self.prefix_path = prefix_path
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = read_image(os.path.join(self.prefix_path, img_path)) / 255.0

        if self.transform:
            img = self.transform(img)
        return img
    
if __name__ == '__main__':
    medical_data = Medical(Path("/home/s/tuyenld/DATA"), Path("/home/s/tuyenld/endoscopy/pretrain.json"))
    train_dataset = PretrainMedical(medical_data.get_train_data(), train=True, 
                                    transform=transforms.Resize((384,384), interpolation=3))
    test_dataset = PretrainMedical(medical_data.get_test_data(), train=False)
    train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=4)
    print(mean_and_std(medical_data.get_train_data(), "/home/s/tuyenld/DATA", "/home/s/tuyenld/mae/configs/meanstd.pth"))
