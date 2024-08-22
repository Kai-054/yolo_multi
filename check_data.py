from pathlib import Path
from ultralytics.data.multi_data import YOLODataset
# from ultralytics.data.dataset import YOLODataset
from torch.utils.data import DataLoader
import torch
import yaml 
from copy import deepcopy

class Check_dataload(YOLODataset):
    def __init__(self, img_path, data=None, task_type="multi"):
        self.data = data
        # self.label_update_count = 0 

        super().__init__(img_path,data=data, task_type=task_type)
    
    # def __getitem__(self, index):
    #     label = self.get_image_and_label(index)
    #     # print("Fetching item at index:", index)
    #     return self.transforms(label)


    # def get_label_info(self, index):
    #         # print("Current index:", index)
    #         # print("Size of self.labels[index]:", (self.labels[index]))
    #         # print("Valid indices:", list(range(len(self.labels[0]))))
    #         if index >= len(self.labels[0]):
    #             raise IndexError(f"Index {index} is out of range for self.labels[0].")
    #         print("Original label:", self.labels[index])
    #         label = deepcopy(self.labels[index])  # out range 
            
    #         label.pop('shape', None)
    #         label['img'], label['ori_shape'], label['resized_shape'] = self.load_image(index)
    #         label['ratio_pad'] = (label['resized_shape'][0] / label['ori_shape'][0],
    #                                   label['resized_shape'][1] / label['ori_shape'][1])
    #         if self.rect:
    #             label['rect_shape'] = self.batch_shapes[self.batch[index]]
    #         label = self.update_labels_info(label)
    #         return label
        
    # def __getitem__(self, idx):
    #     return torch.rand(640, 640, 3), 0
    
# dataset = Check_dataload(img_path="/home/ubuntu/khai202/test_input/train/images")

# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# for images, labels in dataloader:
#     print(images.shape)  
#     print(labels)  # Nhãn
#     break  # Chỉ in ra một batch

with open('/home/ubuntu/khai202/test_input/data.yaml', 'r') as f:
    data = yaml.safe_load(f)
img_path = "/home/ubuntu/khai202/test_input/train/images"
dataset = Check_dataload(img_path, data=data, task_type = "multi" )

# from pathlib import Path
# from ultralytics.data.dataset import YOLODataset
# import yaml

# with open('/home/ubuntu/khai202/test_input/data.yaml', 'r') as f:
#     data = yaml.safe_load(f)
# img_path = "/home/ubuntu/khai202/test_input/train/images"
# print(data)


# dataset2 = YOLODataset(
#     img_path = img_path,
#     data=data,
#     # img_size=640,  # Kích thước hình ảnh tùy chỉnh
#     augment=True,  # Thay đổi tùy theo nhu cầu
#     rect=False,    # Thay đổi tùy theo nhu cầu
# )
# print(datset2[0])
# labels = dataset.get_labels()
# print(labels)

# from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=1, collate_fn=YOLODataset.collate_fn, shuffle=False)
# print(len(dataloader))
# exit()
for batch in dataloader:
    # print("Processing batch")
    # print(len(batch))
    # exit()
    break  
