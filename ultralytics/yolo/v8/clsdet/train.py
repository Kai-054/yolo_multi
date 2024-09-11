from copy import copy

import numpy as np
import torch
import torch.nn as nn

from ultralytics.yolo import v8
from ultralytics.yolo.data import build_dataloader, build_yolo_dataset
from ultralytics.yolo.data.dataloaders.v5loader import create_dataloader
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, RANK, colorstr, ops
from ultralytics.yolo.utils.loss import BboxLoss, FocalLossV1, tversky
from ultralytics.yolo.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.yolo.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.yolo.utils.torch_utils import de_parallel, torch_distributed_zero_first
from ultralytics.nn.tasks import MultiModel
from copy import copy
from ultralytics.yolo.utils.ops import crop_mask, xyxy2xywh, xywh2xyxy
import torch.nn.functional as F
import itertools

class MultitaskTrainer(BaseTrainer):
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a DetectionSegmentationTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides['task'] = 'multi'
        super().__init__(cfg, overrides, _callbacks)
        
    def build_dataset(self, img_path, mode="train", batch=None):
        try:
            gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        except:
            print("bug in here in train.py")
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
       """Construct and return dataloader."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank) 
               
    def preprocess_batch(self, batch):
    """Preprocesses a batch of images, cls_color, and cls_obj."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        batch["cls_color"] = batch["cls_color"].to(self.device)
        batch["cls_obj"] = batch["cls_obj"].to(self.device)

        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(self.args.imgsz * 0.5, self.args.imgsz * 1.5 + self.stride)
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs

        return batch


    def set_model_attributes(self):
        self.model.nc = self.data['nc']  # attach number of classes to model
        self.model.nc_1 = self.data['nc_1']
        self.model.names_2 = self.data['name_2']
        self.model.names_1 = self.data['names_1']  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
            
    def get_model(self, cfg=None, weights=None, verbose=True):
        multi_model = MultiModel(cfg, nc=self.data['nc'], nc_1=self.data['nc_1'], verbose=verbose and RANK == -1)
        if weights:
            multi_model.load(weights)
        return multi_model
        
    def get_validator(self): #return detval and clsval 
        self.loss_names = {"det":['box_loss', 'cls_loss', 'dfl_loss'], "cls":["loss"]}
                return yolo.clsdet.DetclsValidator(   #rewirte val.py
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def criterion(self, preds, batch, name=None, count=None):
        """Compute loss for YOLO prediction and ground-truth."""
        # if 'det' in name:
        #     self.compute_loss = Loss(de_parallel(self.model),count-len(self.data['labels_list']))
        # elif 'seg' in name:
        #     self.compute_loss = SegLoss(de_parallel(self.model), overlap=self.args.overlap_mask, count=count-len(self.data['labels_list']), task_name = name, map=self.data['map'][count])
        return self.compute_loss(preds, batch)

    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys
        
    def progress_string(self):
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )       
        
    def plot_training_samples(self, batch, ni):
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls_color=batch["cls_color"].view(-1),
            cls_obj=batch["cls_obj"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        
        plot_results(file=self.csv, on_plot=self.on_plot)
        
    def plot_training_labels(self):
        
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)
