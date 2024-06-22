import os
import os.path as osp
import torch
import json
import numpy as np
from PIL import Image
import cv2
import albumentations as A

class Data:
    def __init__(self, metadata_file):
        self.train_samples = [] # [type for 4 tasks, image path, mask path, label for task classification]
        self.val_samples = []
        
        with open(metadata_file) as f:
            metadata = json.load(f)['metadata']

        for key in metadata:
            type = metadata[key]['type']
            if type == 'seg':
                self.process_segmentation(metadata, key)
            elif type == 'cls':
                self.process_classification(metadata, key)
                
    def process_segmentation(self, metadata_, key):
        metadata = metadata_[key]
        with open(metadata_["hp"]['annotation']) as f:
            hp_data = json.load(f)

        with open(metadata['annotation']) as f:
            data = json.load(f)

        if key == 'polyp':
          for name in data:
            if name == "train":
              for img_path, mask_path in zip(data[name]["images"], data[name]["masks"]):
                self.train_samples.append([metadata["label"], img_path, mask_path, None])
            elif name == "test":
              for img_path, mask_path in zip(data[name]["images"], data[name]["masks"]):
                self.val_samples.append([metadata["label"], img_path, mask_path, None])
        elif key == 'damage':
          for name in data:
            for name2 in data[name]:
              if name2 == "train":
                for img_path, mask_path in zip(data[name][name2]["images"], data[name][name2]["masks"]):
                  if name != "viem_da_day_20230620":
                      self.train_samples.append([metadata["label"][name], img_path, mask_path, None])
                  else:
                    for e in hp_data["train"]:
                      if e["image"] == img_path:
                        self.train_samples.append([metadata["label"][name], img_path, mask_path, e["label"]])
                        break
              elif name2 == "test":
                for img_path, mask_path in zip(data[name][name2]["images"], data[name][name2]["masks"]):
                  if name != "viem_da_day_20230620":
                      self.val_samples.append([metadata["label"][name], img_path, mask_path, None])
                  else:
                    for e in hp_data["test_positive"]["images"]:
                      if e == img_path:
                        self.val_samples.append([metadata["label"][name], img_path, mask_path, hp_data["test_positive"]["label"]])
                        break
                    for e in hp_data["test_negative"]["images"]:
                      if e == img_path:
                        self.val_samples.append([metadata["label"][name], img_path, mask_path, hp_data["test_negative"]["label"]])
                        break
    
    def process_classification(self, metadata_, key):
        metadata = metadata_[key]
        with open(metadata['annotation']) as f:
            data = json.load(f)
        if key == 'position':
            for name in data:
                for e in data[name]["train"]:
                    self.train_samples.append([metadata["label"], e, None, metadata["cls_label"][name]])
                for e in data[name]["test"]:
                    self.val_samples.append([metadata["label"], e, None, metadata["cls_label"][name]])

class MultiDataset(torch.utils.data.Dataset):
    def __init__(self, data, prefix_path, img_size, transform=None):
        super(MultiDataset, self).__init__()
        self.prefix_path = prefix_path
        self.transform = transform
        self.data = data
        self.img_size = img_size
        self.t = A.Resize(img_size, img_size)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        type, img_path, mask_path, cls_label = self.data[idx]
        
        image = Image.open(osp.join(self.prefix_path, img_path))
        image = np.array(image)
        
        if mask_path is not None:
            mask = cv2.imread(osp.join(self.prefix_path, mask_path), 0)
        else:
            mask = np.zeros_like(image[:,:,0])
        
        if cls_label is None:
            cls_label = -1
            
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
          
        if image.shape != (self.img_size, self.img_size, 3):
            augmented = self.t(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        image = image.astype('float32') / 255
        image = image.transpose((2, 0, 1))

        mask = mask[:,:,np.newaxis]
        mask = mask.astype('float32') / 255
        mask = mask.transpose((2, 0, 1))
        
        return type, image, mask, cls_label
