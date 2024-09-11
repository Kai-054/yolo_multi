import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from multiprocessing.pool import ThreadPool
import posixpath
from ultralytics.yolo.data import build_dataloader, build_yolo_dataset
# from ultralytics.yolo.data.dataloaders.v5loader import create_dataloader
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, colorstr, ops, NUM_THREADS
from ultralytics.yolo.utils.checks import check_requirements
from ultralytics.yolo.utils.metrics import ConfusionMatrix,ClassifyMetrics, DetMetrics, box_iou, SegmentMetrics, mask_iou
from ultralytics.yolo.utils.plotting import output_to_target, plot_images, Annotator, Colors
from ultralytics.yolo.utils.torch_utils import de_parallel
import torch.nn as nn
import math
import contextlib

class DetclsValidator(BaseValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = 'multi'
        self.metrics = []
        self.class_map = [] 
        # if add lables_list in file yaml label can turn on 
        # try:
        #     for name in dataloader.dataset.data['labels_list']:
        #         if 'det' in name:
        #             self.metrics.append(DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot))
        #         if "cls" in name:
        #             self.metrics.append(ClassifyMetric(save_dir=self.save_dir, on_plot=self.on_plot))
        # except:
        #     print("bug in here val.py in clsdet")
        self.metrics_det = DetMetrics(save_dir=self.save_dir, on_plot= self.on_plot)
        self.metrics_cls = ClassifyMetrics()
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls_obj","cls_color", "bboxes"]: #khai have a some problem in batch_idx 
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = [
                torch.cat([batch["cls_obj"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                for i in range(nb)
            ]
            self.lb_1 = [
                torch.cat([batch["cls_color"][batch["batch_idx"] == i]], dim=-1)
                for i in range(nb)
            ]
        return batch
    
    def init_metrics(self, model):
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        
    def get_desc_det(self):
        return ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'Box(P', 'R', 'mAP50', 'mAP50-95)')

    def get_desc_cls(self):
        return ("%22s" + "%11s" * 2) % ("classes", "top1_acc", "top5_acc")
   
    def get_desc_clsdet(self):
        return("%22s" + "%11s" * 8) % ('Class', 'Images', 'Instances', 'Box(P', 'R', 'mAP50', 'mAP50-95)',"classes", "top1_acc", "top5_acc")
    
    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
        )
        return preds
        
        
    def update_metrics(self, preds, batch, task_name=None):
            if self.args.combine_class:
                for si, pred in enumerate(preds):
                    idx = batch['batch_idx'] == si
                    cls = batch['cls'][idx]
                    cls = self.replace_elements_in_column(cls,self.args.combine_class,0)
                    bbox = batch['bboxes'][idx]
                    nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
                    shape = batch['ori_shape'][si]
                    correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
                    self.seen[task_name] += 1

                    if npr == 0:
                        if nl:
                            self.stats[task_name].append(
                                (correct_bboxes, *torch.zeros((2, 0), device=self.device), cls.squeeze(-1)))
                            if self.args.plots:
                                self.confusion_matrix[task_name].process_batch(detections=None, labels=cls.squeeze(-1))
                        continue

                    # Predictions
                    if self.args.single_cls:
                        pred[:, 5] = 0
                    predn = pred.clone()
                    predn = self.replace_elements_in_column(predn,self.args.combine_class,5)
                    ops.scale_boxes(batch['img'][si].shape[1:], predn[:, :4], shape,
                                    ratio_pad=batch['ratio_pad'][si])  # native-space pred

                    # Evaluate
                    if nl:
                        height, width = batch['img'].shape[2:]
                        tbox = ops.xywh2xyxy(bbox) * torch.tensor(
                            (width, height, width, height), device=self.device)  # target boxes
                        ops.scale_boxes(batch['img'][si].shape[1:], tbox, shape,
                                        ratio_pad=batch['ratio_pad'][si])  # native-space labels
                        labelsn = torch.cat((cls, tbox), 1)  # native-space labels
                        correct_bboxes = self._process_batch_det(predn, labelsn)
                        # TODO: maybe remove these `self.` arguments as they already are member variable
                        if self.args.plots:
                            self.confusion_matrix[task_name].process_batch(predn, labelsn)
                    self.stats[task_name].append(
                        (correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)

                    # Save
                    if self.args.save_json:
                        self.pred_to_json(predn, batch['im_file'][si])
                    if self.args.save_txt:
                        file = self.save_dir / 'labels' / f'{Path(batch["im_file"][si]).stem}.txt'
                        self.save_one_txt(predn, self.args.save_conf, shape, file)
            else:
                for si, pred in enumerate(preds):
                    idx = batch['batch_idx'] == si
                    cls = batch['cls'][idx]
                    bbox = batch['bboxes'][idx]
                    nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
                    shape = batch['ori_shape'][si]
                    correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
                    self.seen[task_name] += 1

                    if npr == 0:
                        if nl:
                            self.stats[task_name].append((correct_bboxes, *torch.zeros((2, 0), device=self.device), cls.squeeze(-1)))
                            if self.args.plots:
                                self.confusion_matrix[task_name].process_batch(detections=None, labels=cls.squeeze(-1))
                        continue

                    # Predictions
                    if self.args.single_cls:
                        pred[:, 5] = 0
                    predn = pred.clone()
                    ops.scale_boxes(batch['img'][si].shape[1:], predn[:, :4], shape,
                                    ratio_pad=batch['ratio_pad'][si])  # native-space pred

                    # Evaluate
                    if nl:
                        height, width = batch['img'].shape[2:]
                        tbox = ops.xywh2xyxy(bbox) * torch.tensor(
                            (width, height, width, height), device=self.device)  # target boxes
                        ops.scale_boxes(batch['img'][si].shape[1:], tbox, shape,
                                        ratio_pad=batch['ratio_pad'][si])  # native-space labels
                        labelsn = torch.cat((cls, tbox), 1)  # native-space labels
                        correct_bboxes = self._process_batch_det(predn, labelsn)
                        # TODO: maybe remove these `self.` arguments as they already are member variable
                        if self.args.plots:
                            self.confusion_matrix[task_name].process_batch(predn, labelsn)
                    self.stats[task_name].append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)

                    # Save
                    if self.args.save_json:
                        self.pred_to_json(predn, batch['im_file'][si])
                    if self.args.save_txt:
                        file = self.save_dir / 'labels' / f'{Path(batch["im_file"][si]).stem}.txt'
                        self.save_one_txt(predn, self.args.save_conf, shape, file)
                        
    def update_metrics_cls (self, preds, batch):
        n5 = min(len(self.names), 5)
        self.pred.append(preds.argsort(1, descending=True)[:, :n5].type(torch.int32).cpu())
        self.targets.append(batch["cls"].type(torch.int32).cpu())
    
    def finalize_metric(self, *args, **kwargs):
        """Set final values for metrics speed and confusion matrix."""
        self.confusion_matrix.process_cls_preds(self.pred, self.targets)
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )
                
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix 
        self.metrics.save_dir = self.save_dir
        
    def get_stats(self):
    """Returns metrics statistics and results dictionary."""
    # Chuyển các tensor trong self.stats thành mảng NumPy
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}# Tính số lượng mục tiêu trên mỗi lớp
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)# Tính số lượng mục tiêu trên mỗi hình ảnh
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)# Loại bỏ khóa target_img khỏi stats
        stats.pop("target_img", None)    # Nếu có dữ liệu và có bất kỳ True Positive nào, xử lý thống kê
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)# Trả về từ điển kết quả từ self.metrics
        return self.metrics.results_dict
    
    def print_results(self):
        
   

        