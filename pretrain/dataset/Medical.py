import os
import json
from pathlib import Path
import PIL

import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def build_transforms(input_size=224, is_train=False, meanstd={'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD}):    
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop((input_size, input_size), scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=meanstd["mean"], std=meanstd["std"]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=3),
            transforms.ToTensor(),
            ])
        
    return transform

def unnormalize(img, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, max_pixel_value=255.0):
    img = (img * std  + mean) * max_pixel_value
    return np.clip(np.round(img), 0, 255).astype(np.uint8)

def mean_and_std(train_data, prefix_path, meanstd_file):
    """
    Compute the mean and std of the dataset
    Args:
            img_dir: the directory of the images
            meanstd_file: the file to save the mean and std
    """
    import cupy as cp
    # BEGIN CODE
    if os.path.isfile(meanstd_file):
        meanstd = torch.load(meanstd_file)
    else:
        means, stds = [], []
        for i, img_path in enumerate(train_data):
            print(f"Processing image {i+1}/{len(train_data)}")
            img = PIL.Image.open(Path(prefix_path) / img_path)
            img = np.array(img) / 255.0
            img = cp.asarray(img)
            mean = cp.mean(img, axis=(0, 1))
            std = cp.std(img, axis=(0, 1))
            
            means.append(mean.get())  # Transfer result back to CPU
            stds.append(std.get())    # Transfer result back to CPU
        
        mean = np.mean(means, axis=0)
        std = np.mean(stds, axis=0)
        meanstd = {'mean': mean, 'std': std}
        torch.save(meanstd, meanstd_file)
        print("Mean and std: ", meanstd)
    # END CODE
    return meanstd

class PretrainMedical(Dataset):
    def __init__(self, data=None, meanstd={'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD}, prefix_path=None, train=True):
        self.data = data
        
        self.meanstd = meanstd

        if train:
            self.transform = build_transforms(is_train=True, meanstd=self.meanstd)
        else:
            self.transform = build_transforms(is_train=False, meanstd=self.meanstd)

        self.prefix_path = prefix_path
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        img_path = self.data[idx]
        path = Path(self.prefix_path) / img_path
        img = PIL.Image.open(path)

        if self.transform:
            img = self.transform(img)
        return img
    
if __name__ == '__main__':
    # meanstd = torch.load("/mnt/tuyenld/mae/configs/meanstd.pth")
    # # medical_data = Medical(Path("/home/s/tuyenld/DATA"), Path("/home/s/tuyenld/endoscopy/pretrain.json"))
    # test_dataset = PretrainMedical(train=False, 
    #                                 json_file="/mnt/tuyenld/mae/configs/test.json",
    #                                 meanstd_file="/mnt/tuyenld/mae/configs/meanstd.pth",
    #                                 prefix_path="/home/s/DATA")
    # # test_dataset = PretrainMedical(medical_data.get_test_data(), train=False)
    # # train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=4)
    # # print(mean_and_std(medical_data.get_train_data(), "/home/s/tuyenld/DATA", "/home/s/tuyenld/mae/configs/meanstd.pth"))
    # for data in test_dataset:
    #     print(data)
    # Multi processing calculate mean and std
    # import multiprocessing
    # from functools import partial
    # from multiprocessing import Pool
    # from itertools import repeat
    
    data = json.load(open('/workspace/endoscopy/pretrain.json'))
    train_data = data["train"]
    prefix_path = "/workspace/DATA2"
    meanstd_file = "/workspace/DATA2/meanstd_with_extract.pth"
    mean_and_std(train_data, prefix_path, meanstd_file)

    # num_workers = 48
    # pool = Pool(num_workers)
    # meanstd = pool.map(partial(mean_and_std, prefix_path=prefix_path, meanstd_file=meanstd_file), repeat(train_data, num_workers))
    # pool.join()
    # pool.close()
    
    
