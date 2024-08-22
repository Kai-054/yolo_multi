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
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        
    def get_validator(self): #return detval and clsval 
        self.loss_names = {"det":['box_loss', 'cls_loss', 'dfl_loss'], "cls":["loss"]}
                return yolo.clsdet.DetclsValidator(   #rewirte val.py
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
        
    def label_loss_items(self, loss_items=None, prefix="train"):
        
    def progress_string(self):
        
    def plot_training_samples(self, batch, ni):
        fname = self.save_dir / f"train_batch{self.data['labels_list'][task]}{ni}.jpg" if task!=None else self.save_dir / f'train_batch{ni}.jpg'
                if 'det' in self.data['labels_list'][task]:
                    plot_images(images=batch['img'],
                        batch_idx=batch['batch_idx'],
                        cls=batch['cls'].squeeze(-1),
                        bboxes=batch['bboxes'],
                        paths=batch['im_file'],
                        fname=fname,
                        on_plot=self.on_plot)
                elif 'cls' :

        
    def plot_metrics(self):
        
    def plot_training_labels(self):