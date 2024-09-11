import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from pathlib import Path
from ultralytics.yolo.utils.instance import Instances
from multiprocessing.pool import ThreadPool
from itertools import repeat
import traceback
from ultralytics.yolo.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr, is_dir_writeable
from .augment import Compose, Format, Instances, LetterBox, classify_transforms, v8_transforms
from .base import BaseDataset
from .utils import HELP_URL, LOGGER, get_hash, img2label_paths,label2img_paths, verify_image_mlt_label
# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
DATASET_CACHE_VERSION = "1.0.3"

class YOLODataset(BaseDataset):
    def __init__(self, *args, data=None, task_type=False, **kwargs):
        """Initializes the YOLODataset with optional configurations for segments and keypoints."""
        self.data = data
        self.task_type = task_type 
        super().__init__(*args, **kwargs)

    def cache_labels(self, path=Path("./labels.cache")):
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_mlt_label,
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(len(self.data["name_1"])), #name logo
                    repeat(len(self.data["name_2"])), #name types car
                    repeat(nkpt),
                    repeat(ndim),
                ),
            )

            pbar = TQDM(results, desc=desc, total=total)
            # for result in pbar:
            #     print(result)
            for im_file, lb, shape, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls_color": lb[:, 0:1],
                            "cls_obj": lb[:, 1:2],  # n, 1
                            "bboxes": lb[:, 2:],  # n, 4
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x)
        return x

    def get_labels(self):
        """Trả về từ điển nhãn cho đào tạo Yolo."""
        self.label_files = img2label_paths(self.im_files)
        # self.images_files = label2img_paths(self.im_files)  # custom
        
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in (-1, 0):
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        
        if not labels:
            LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}")
            
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files
        self.label_file = [lb["im_file"] for lb in labels]  # fix to match BaseDataset
        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls_color"]), len(lb["cls_obj"]), len(lb["bboxes"])) for lb in labels)
        
        if not any(lengths):
            LOGGER.warning(f"khai WARNING ⚠️ No valid label data found in {cache_path}, training may not work correctly. {HELP_URL}")
        
        len_cls_color, len_cls_obj, len_boxes = (sum(x) for x in zip(*lengths))
        if len_cls_obj == 0:
            LOGGER.warning(f"WARNING ⚠️ No object labels found in {cache_path}, training may not work correctly. {HELP_URL}")
            
        if len_cls_color == 0:
            LOGGER.warning(f"WARNING ⚠️ No color labels found in {cache_path}, training may not work correctly. {HELP_URL}")

        return labels

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """
        Custom your label format here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        # self.label_update_count += 1 
        # print(f"Số lần update_labels_info được gọi: {self.label_update_count}")  # In bộ đếm
        # traceback.print_stack()

        bboxes = label.pop("bboxes")
        # print("khai2", bboxes)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        label["instances"] = Instances(bboxes, bbox_format=bbox_format, normalized=normalized)
        # print("khai3", label["instances"])
        # print(f"Số lần update_labels_info được gọi: {self.label_update_count}")  
        return label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        # print("Batch:", batch)  # In ra toàn bộ batch
        # print("Type of batch[0]:", type(batch[0]))
        new_batch = {}
        new_batch_list = []
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            # print("values", values)
            # print('i=',i)
            # print('k=',k)
            
            if k == "img":
                value = torch.stack(value, 0)
                # print ("value of img", value)
                try:
                    value = torch.stack(value, 0)
                    # print ("value of img", value)
                except Exception as e:
                    print(f"Error stacking 'img' tensors: {[v.shape for v in value]}")
                    print(e)
            elif k in ["bboxes", "cls_color", "cls_obj"]:
                try:
                    # print(f"Concatenating '{k}' tensors: {[v.shape for v in value]}")
                    value = torch.cat(value, 0)
                    # print ("value of k(bboxes", "cls_color", "cls_obj)", value)

                except Exception as e:
                    print(f"Error concatenating '{k}' tensors: {[v.shape for v in value]}")
                    print(e)
                    
            new_batch[k] = value
            
        # print("Batch before update:", new_batch)
            # if "batch_idx" not in new_batch:
            #     print("No batch_idx in the batch, adding batch_idx.")
            #     new_batch["batch_idx"] = torch.arange(len(batch)).unsqueeze(1)
                
            # print("Batch after adding batch_idx:", new_batch)  
                 
        #     if "batch_idx" in new_batch:
        #         print("Batch indices before update:", new_batch["batch_idx"])
        #         new_batch["batch_idx"] = list(new_batch["batch_idx"])
        #         for i in range(len(new_batch["batch_idx"])):
        #             new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        #         new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        #     else:
        #         print("No batch_idx in the batch")
            
        # new_batch["batch_idx"] = list(new_batch["batch_idx"])
        # new_batch["batch_idx"] = list(new_batch.get("batch_idx", []))
        # for i in range(len(new_batch["batch_idx"])):
        #     new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        # new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        # print("Batch indices after update:", new_batch["batch_idx"]) 
        return new_batch
    
def save_dataset_cache_file(prefix, path, x):
    """Save an Ultralytics dataset *.cache dictionary x to path."""
    x["version"] = DATASET_CACHE_VERSION  # add cache version
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()  # remove *.cache file if exists
        np.save(str(path), x)  # save cache for next time
        path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
        LOGGER.info(f"{prefix}New cache created: {path}")
    else:
        LOGGER.warning(f"{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.")
        
def load_dataset_cache_file(path):
    """Load an Ultralytics *.cache dictionary from path."""
    import gc

    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache = np.load(str(path), allow_pickle=True).item()  # load dict
    gc.enable()
    return cache
