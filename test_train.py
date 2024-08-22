from ultralytics import YOLO


# Load a model
model = YOLO('ultralytics/models/v8/yolov8_multi.yaml', task='multi')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
model.train(data='ultralytics/datasets/multi_data.yaml', batch=12, epochs=300, imgsz=(640,640), device=[0,1,2], name='yolopm', val=True, task='multi',classes=[2,3,4,9,10,11],combine_class=[2,3,4,9],single_cls=True)
