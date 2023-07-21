
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
import torchvision.transforms as transforms  

IMAGE_FOLDER = "./20220804_1w" 
class MyDataSet(Dataset):  
    def __init__(self, data, soft_labels_filename=None, transforms=None):  
        self.data = data  
        self.transforms = transforms  
  
    def __getitem__(self, index):  
        # Initialize transform and normalize  
        transform = transforms.Compose([  
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),  
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
        ])  
  
        # Read images  
        folder_path = os.path.join(IMAGE_FOLDER, self.data.iloc[index, 0],'torch')  
        image_filenames = sorted(os.listdir(folder_path))[:25]  
        images = []  
        for img_name in image_filenames:  
            image_path = os.path.join(folder_path, img_name)  
            image = Image.open(image_path)  
            image = transform(image)  
            images.append(image)  
  
        # Stack images  
        images_tensor = torch.stack(images)  
  
        label = self.data.iloc[index, 1]  
  
        return images_tensor, label  
    def __len__(self):
        return len(self.data)
    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels