from ultralytics.nn.tasks import parse_model, yaml_model_load, MultiModel
import torch 

# img_pth = "/home/ubuntu/khai202/data/Van/27_jpg.rf.0ff82c13be0af1f9f1615b754843e117.jpg"
model_pth = "/home/ubuntu/khai202/yolo_multi/ultralytics/models/v8/yolov8_multi.yaml"
model_dict = yaml_model_load(model_pth)
#  = img_pth 
input_channels = 3
model = MultiModel(model_dict, input_channels )
# x = torch.randn(1, input_channels, 640, 640)
# output = model(x)
# print("Output:", output)

# Kiểm tra kết quả
print(model_dict)  # In ra cấu trúc của mô hình
print("Layers to save:", save)  # In ra danh sách các tầng cần lưu