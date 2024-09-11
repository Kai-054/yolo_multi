import torch 
import torchvision 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.tasks import MultiModel

data = '/home/ubuntu/khai202/yolo_multi/ultralytics/datasets/multi_data.yaml'
model = YOLO('/home/ubuntu/khai202/yolo_multi/ultralytics/models/v8/yolov8_multi.yaml', task='multi')  # build a new model from YAML
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


batch=100
epochs=10
# Split the training set into training and validation sets 
val_percent = 0.2 # percentage of the data used for validation 
val_size = int(val_percent * len(train_dataset)) 
train_size = len(train_dataset) - val_size 
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, 
														[train_size, 
															val_size]) 

# Create DataLoaders for the training and validation sets 
train_loader = torch.utils.data.DataLoader(train_dataset, 
										batch_size=batch_size, 
										shuffle=True, 
										pin_memory=True) 
val_loader = torch.utils.data.DataLoader(val_dataset, 
										batch_size=batch_size, 
										shuffle=False, 
										pin_memory=True) 

def get_model(self, cfg=None, weights=None, verbose=True):
        multi_model = MultiModel(cfg, nc=self.data['nc'], nc_1=self.data['nc_1'], verbose=verbose and RANK == -1)
        if weights:
            multi_model.load(weights)
        return multi_model
        
losses = [] 
accuracies = [] 
val_losses = [] 
val_accuracies = [] 
# Train the model 
for epoch in range(num_epochs): 
	for i, (images, labels) in enumerate(train_loader): 
		# Forward pass 
		images=images.to(device) 
		labels=labels.to(device) 
		outputs = model(images) 
		loss = criterion(outputs, labels) 
		
		# Backward pass and optimization 
		optimizer.zero_grad() 
		loss.backward() 
		optimizer.step() 

		_, predicted = torch.max(outputs.data, 1) 
	acc = (predicted == labels).sum().item() / labels.size(0) 
	accuracies.append(acc) 
	losses.append(loss.item()) 
		
	# Evaluate the model on the validation set 
	val_loss = 0.0
	val_acc = 0.0
	with torch.no_grad(): 
		for images, labels in val_loader: 
			labels=labels.to(device) 
			images=images.to(device) 
			outputs = model(images) 
			loss = criterion(outputs, labels) 
			val_loss += loss.item() 
			
			_, predicted = torch.max(outputs.data, 1) 
		total = labels.size(0) 
		correct = (predicted == labels).sum().item() 
		val_acc += correct / total 
		val_accuracies.append(acc) 
		val_losses.append(loss.item()) 
	
			
	print('Epoch [{}/{}],Loss:{:.4f},Validation Loss:{:.4f},Accuracy:{:.2f},Validation Accuracy:{:.2f}'.format( 
		epoch+1, num_epochs, loss.item(), val_loss, acc ,val_acc))



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

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
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
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
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
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

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
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

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