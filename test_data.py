import os
import yaml
import torch
from torch.utils.data import DataLoader
from ultralytics.yolo.data.dataset import YOLODataset
import matplotlib.pyplot as plt
import torchvision.transforms as T
from pathlib import Path


class Check_dataload(YOLODataset):
    def __init__(self, img_path, data=None, task_type="multi"):
        self.data = data
        # self.label_update_count = 0

        super().__init__(img_path, data=data, task_type=task_type)


with open(r'C:\Users\User\Desktop\yolo_multi\ultralytics\datasets\multi_data.yaml', 'r') as f:
    data = yaml.safe_load(f)
img_path = r"C:\Users\User\Desktop\data_multi\train\images"
dataset = Check_dataload(img_path, data=data, task_type="multi")


dataloader = DataLoader(
    dataset,
    batch_size=1,
    collate_fn=YOLODataset.collate_fn,
    shuffle=False)
# print(len(dataloader))
# exit()
for batch in dataloader:
    # print("Processing batch")
    # print(len(batch))
    # exit()
    break
