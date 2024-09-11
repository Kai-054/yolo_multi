import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from ultralytics import YOLO
from ultralytics.nn.tasks import MultiModel
from ultralytics.yolo.data import YOLODataset
import yaml


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_path = r"C:\Users\User\Desktop\data_multi\train\images"
data = r'C:\Users\User\Desktop\yolo_multi\ultralytics\datasets\multi_data.yaml'
yolo = YOLO(
    r'C:\Users\User\Desktop\yolo_multi\ultralytics\models\v8\yolov8_multi.yaml',
    task='multi')
# print(dir(yolo))
# exit()
# yolo = yolo.to(device)
model = yolo.model.to(device)
print(next(model.parameters()).is_cuda)
# exit()


batch_size = 1
num_epochs = 10
train_dataset = YOLODataset(data=data, img_path=img_path)
# batch = next(iter(train_dataset))
# print("batch_train_dataset",batch)
# exit()

#--------------------------------------------------------------------
# data_transforms = T.Compose([
#     T.Resize((640, 640)),  # Resize image to 640x640
#     T.ToTensor(),          # Convert PIL Image or numpy array to tensor
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using mean and std
# ])

# train_dataset = YOLODataset(data=data, img_path=img_path)
# transforms = data_transforms

# def my_collate_fn(batch):
#     targets = []
#     imgs = []
#     for sample in batch:
#         imgs.append(sample[0])
#         targets.append(torch.FloatTensor(sample[1]))
#     #batch_size (so luong buc anh)
#     imgs = torch.stack(imgs, dim=0)
#     return imgs, targets 
#-----------------------------------------------------------------------  
    
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           collate_fn=train_dataset.collate_fn)

# batch = next(iter(train_loader))
# print(batch)
# exit()
with open(data, 'r') as file:
    data_config = yaml.safe_load(file)


# print("Data config:", data_config)

train_dataset = YOLODataset(data=data, img_path=img_path)

val_dataset = YOLODataset(data=data, img_path=img_path)
val_percent = 0.2  # percentage of the data used for validation
val_size = int(val_percent * len(train_dataset))
train_size = len(train_dataset) - val_size

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,
                                                           [train_size,
                                                            val_size])
val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False
                                         )
# batch = next(iter(val_loader))
# print("value of batch ",batch)
# exit()

def get_model(self, cfg=None, weights=None, verbose=True):
    multi_model = MultiModel(
        cfg,
        nc=self.data['nc'],
        nc_1=self.data['nc_1'],
        verbose=verbose and RANK == -1)
    if weights:
        multi_model.load(weights)
    return multi_model


losses = []
accuracies = []
val_losses = []
val_accuracies = []
# Train the model
for epoch in range(num_epochs):
    model.train()
    for i, batch in enumerate(train_loader):
        # print(f"********************Batch value {i}:", batch)
        # exit()
        # print(i)
        # Forward pass
        image = batch['img'].float()/255
        images = image.to(device)
        # print(f"Input tensor type: {images.type()}")
        # print(f"Input tensor dtype: {images.dtype}")
        # print(f"Input tensor device: {images.device}")
        # exit()
        cls_obj = batch['cls_obj'].to(device)
        cls_color = batch['cls_color'].to(device)
        bboxes = batch['bboxes'].to(device)
        outputs = model(images)
        # print(outputs)
        # exit()
        
        loss = criterion(outputs, cls_obj, cls_color, bboxes)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs['cls_obj'].data, 1)
    acc = (predicted == labels).sum().item() / labels.size(0)
    accuracies.append(acc)
    losses.append(loss.item())

    # Evaluate the model on the validation set
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images = batch['img'].to(device)
            cls_obj = batch['cls_obj'].to(device)
            cls_color = batch['cls_color'].to(device)
            bboxes = batch['bboxes'].to(device)
            outputs = model(images)

            loss = criterion(outputs, cls_obj, cls_color, bboxes)
            val_loss += loss.item()

            _, predicted = torch.max(outputs['cls_obj'].data, 1)
        total = cls_obj.size(0)
        correct = (predicted == cls_obj).sum().item()
        val_acc += correct / total
        val_accuracies.append(acc)
        val_losses.append(loss.item())

    print(
        'Epoch [{}/{}],Loss:{:.4f},Validation Loss:{:.4f},Accuracy:{:.2f},Validation Accuracy:{:.2f}'.format(
            epoch +
            1,
            num_epochs,
            loss.item(),
            val_loss,
            acc,
            val_acc))


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(
                batch_size,
                counts.max(),
                ne - 1,
                device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(
                b,
                a,
                4,
                c //
                4).softmax(3).matmul(
                self.proj.type(
                    pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(
            feats[0].shape[0], self.no, -1) for xi in feats], 2).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:],
                             device=self.device,
                             dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(
            self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(
            anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores,
        # target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(
            dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items
