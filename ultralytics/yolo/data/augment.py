import math
import random
from copy import deepcopy
from typing import Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from ultralytics.yolo.utils.instance import Instances
from ultralytics.yolo.utils import LOGGER, colorstr




DEFAULT_MEAN = (0.0, 0.0, 0.0)
DEFAULT_STD = (1.0, 1.0, 1.0)
DEFAULT_CROP_FRACTION = 1.0

class Compose:
    """Class for composing multiple image transformations."""

    def __init__(self, transforms):
        """Initializes the Compose object with a list of transforms."""
        self.transforms = transforms

    def __call__(self, data):
        """Applies a series of transformations to input data."""
        for t in self.transforms:
            data = t(data)
        return data

    def append(self, transform):
        """Appends a new transform to the existing list of transforms."""
        self.transforms.append(transform)

    def tolist(self):
        """Converts the list of transforms to a standard Python list."""
        return self.transforms

    def __repr__(self):
        """Returns a string representation of the object."""
        return f"{self.__class__.__name__}({', '.join([f'{t}' for t in self.transforms])})"
    
class RandomFlip:
    """
    Applies a random horizontal or vertical flip to an image with a given probability.

    This class performs random image flipping and updates corresponding instance annotations such as
    bounding boxes and keypoints.

    Attributes:
        p (float): Probability of applying the flip. Must be between 0 and 1.
        direction (str): Direction of flip, either 'horizontal' or 'vertical'.
        flip_idx (array-like): Index mapping for flipping keypoints, if applicable.

    Methods:
        __call__: Applies the random flip transformation to an image and its annotations.

    Examples:
        >>> transform = RandomFlip(p=0.5, direction='horizontal')
        >>> result = transform({"img": image, "instances": instances})
        >>> flipped_image = result["img"]
        >>> flipped_instances = result["instances"]
    """

    def __init__(self, p=0.5, direction="horizontal", flip_idx=None) -> None:
        # Initializes the RandomFlip class with probability and direction.

        # This class applies a random horizontal or vertical flip to an image with a given probability.
        # It also updates any instances (bounding boxes, keypoints, etc.) accordingly.

        # Args:
        #     p (float): The probability of applying the flip. Must be between 0 and 1.
        #     direction (str): The direction to apply the flip. Must be 'horizontal' or 'vertical'.
        #     flip_idx (List[int] | None): Index mapping for flipping keypoints, if any.

        # Raises:
        #     AssertionError: If direction is not 'horizontal' or 'vertical', or if p is not between 0 and 1.

        # Examples:
        #     >>> flip = RandomFlip(p=0.5, direction='horizontal')
        #     >>> flip = RandomFlip(p=0.7, direction='vertical', flip_idx=[1, 0, 3, 2, 5, 4])
        assert direction in {"horizontal", "vertical"}, f"Support direction `horizontal` or `vertical`, got {direction}"
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."

        self.p = p
        self.direction = direction
        self.flip_idx = flip_idx

    def __call__(self, labels):
        # Applies random flip to an image and updates any instances like bounding boxes or keypoints accordingly.

        # This method randomly flips the input image either horizontally or vertically based on the initialized
        # probability and direction. It also updates the corresponding instances (bounding boxes, keypoints) to
        # match the flipped image.

        # Args:
        #     labels (Dict): A dictionary containing the following keys:
        #         'img' (numpy.ndarray): The image to be flipped.
        #         'instances' (ultralytics.utils.instance.Instances): An object containing bounding boxes and
        #             optionally keypoints.

        # Returns:
        #     (Dict): The same dictionary with the flipped image and updated instances:
        #         'img' (numpy.ndarray): The flipped image.
        #         'instances' (ultralytics.utils.instance.Instances): Updated instances matching the flipped image.

        # Examples:
        #     >>> labels = {'img': np.random.rand(640, 640, 3), 'instances': Instances(...)}
        #     >>> random_flip = RandomFlip(p=0.5, direction='horizontal')
        #     >>> flipped_labels = random_flip(labels)
        
        img = labels["img"]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xywh")
        h, w = img.shape[:2]
        h = 1 if instances.normalized else h
        w = 1 if instances.normalized else w

        # Flip up-down
        if self.direction == "vertical" and random.random() < self.p:
            img = np.flipud(img)
            instances.flipud(h)
        if self.direction == "horizontal" and random.random() < self.p:
            img = np.fliplr(img)
            instances.fliplr(w)
            # For keypoints
            if self.flip_idx is not None and instances.keypoints is not None:
                instances.keypoints = np.ascontiguousarray(instances.keypoints[:, self.flip_idx, :])
        labels["img"] = np.ascontiguousarray(img)
        labels["instances"] = instances
        return labels


class LetterBox:
    """
    Resize image and padding for detection, instance segmentation, pose.

    This class resizes and pads images to a specified shape while preserving aspect ratio. It also updates
    corresponding labels and bounding boxes.

    Attributes:
        new_shape (tuple): Target shape (height, width) for resizing.
        auto (bool): Whether to use minimum rectangle.
        scaleFill (bool): Whether to stretch the image to new_shape.
        scaleup (bool): Whether to allow scaling up. If False, only scale down.
        stride (int): Stride for rounding padding.
        center (bool): Whether to center the image or align to top-left.

    Methods:
        __call__: Resize and pad image, update labels and bounding boxes.

    Examples:
        >>> transform = LetterBox(new_shape=(640, 640))
        >>> result = transform(labels)
        >>> resized_img = result['img']
        >>> updated_instances = result['instances']
    """

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
        """
        Initialize LetterBox object for resizing and padding images.

        This class is designed to resize and pad images for object detection, instance segmentation, and pose estimation
        tasks. It supports various resizing modes including auto-sizing, scale-fill, and letterboxing.

        Args:
            new_shape (Tuple[int, int]): Target size (height, width) for the resized image.
            auto (bool): If True, use minimum rectangle to resize. If False, use new_shape directly.
            scaleFill (bool): If True, stretch the image to new_shape without padding.
            scaleup (bool): If True, allow scaling up. If False, only scale down.
            center (bool): If True, center the placed image. If False, place image in top-left corner.
            stride (int): Stride of the model (e.g., 32 for YOLOv5).

        Attributes:
            new_shape (Tuple[int, int]): Target size for the resized image.
            auto (bool): Flag for using minimum rectangle resizing.
            scaleFill (bool): Flag for stretching image without padding.
            scaleup (bool): Flag for allowing upscaling.
            stride (int): Stride value for ensuring image size is divisible by stride.

        Examples:
            >>> letterbox = LetterBox(new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32)
            >>> resized_img = letterbox(original_img)
        """
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left

    def __call__(self, labels=None, image=None):
        """
        Resizes and pads an image for object detection, instance segmentation, or pose estimation tasks.

        This method applies letterboxing to the input image, which involves resizing the image while maintaining its
        aspect ratio and adding padding to fit the new shape. It also updates any associated labels accordingly.

        Args:
            labels (Dict | None): A dictionary containing image data and associated labels, or empty dict if None.
            image (np.ndarray | None): The input image as a numpy array. If None, the image is taken from 'labels'.

        Returns:
            (Dict | Tuple): If 'labels' is provided, returns an updated dictionary with the resized and padded image,
                updated labels, and additional metadata. If 'labels' is empty, returns a tuple containing the resized
                and padded image, and a tuple of (ratio, (left_pad, top_pad)).

        Examples:
            >>> letterbox = LetterBox(new_shape=(640, 640))
            >>> result = letterbox(labels={'img': np.zeros((480, 640, 3)), 'instances': Instances(...)})
            >>> resized_img = result['img']
            >>> updated_instances = result['instances']
        """
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """
        Updates labels after applying letterboxing to an image.

        This method modifies the bounding box coordinates of instances in the labels
        to account for resizing and padding applied during letterboxing.

        Args:
            labels (Dict): A dictionary containing image labels and instances.
            ratio (Tuple[float, float]): Scaling ratios (width, height) applied to the image.
            padw (float): Padding width added to the image.
            padh (float): Padding height added to the image.

        Returns:
            (Dict): Updated labels dictionary with modified instance coordinates.

        Examples:
            >>> letterbox = LetterBox(new_shape=(640, 640))
            >>> labels = {'instances': Instances(...)}
            >>> ratio = (0.5, 0.5)
            >>> padw, padh = 10, 20
            >>> updated_labels = letterbox._update_labels(labels, ratio, padw, padh)
        """
        labels["instances"].convert_bbox(format="xyxy")
        print("bug_augment ",labels["instances"].convert_bbox(format="xyxy"))
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels

class RandomHSV:
    """
    Randomly adjusts the Hue, Saturation, and Value (HSV) channels of an image.

    This class applies random HSV augmentation to images within predefined limits set by hgain, sgain, and vgain.

    Attributes:
        hgain (float): Maximum variation for hue. Range is typically [0, 1].
        sgain (float): Maximum variation for saturation. Range is typically [0, 1].
        vgain (float): Maximum variation for value. Range is typically [0, 1].

    Methods:
        __call__: Applies random HSV augmentation to an image.

    Examples:
        >>> import numpy as np
        >>> from ultralytics.data.augment import RandomHSV
        >>> augmenter = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
        >>> image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        >>> labels = {"img": image}
        >>> augmented_labels = augmenter(labels)
        >>> augmented_image = augmented_labels["img"]
    """

    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5) -> None:
        # """
        # Initializes the RandomHSV object for random HSV (Hue, Saturation, Value) augmentation.

        # This class applies random adjustments to the HSV channels of an image within specified limits.

        # Args:
        #     hgain (float): Maximum variation for hue. Should be in the range [0, 1].
        #     sgain (float): Maximum variation for saturation. Should be in the range [0, 1].
        #     vgain (float): Maximum variation for value. Should be in the range [0, 1].

        # Examples:
        #     >>> hsv_aug = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
        #     >>> augmented_image = hsv_aug(image)
        # """
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, labels):
        """
        Applies random HSV augmentation to an image within predefined limits.

        This method modifies the input image by randomly adjusting its Hue, Saturation, and Value (HSV) channels.
        The adjustments are made within the limits set by hgain, sgain, and vgain during initialization.

        Args:
            labels (Dict): A dictionary containing image data and metadata. Must include an 'img' key with
                the image as a numpy array.

        Returns:
            (None): The function modifies the input 'labels' dictionary in-place, updating the 'img' key
                with the HSV-augmented image.

        Examples:
            >>> hsv_augmenter = RandomHSV(hgain=0.5, sgain=0.5, vgain=0.5)
            >>> labels = {'img': np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)}
            >>> hsv_augmenter(labels)
            >>> augmented_img = labels['img']
        """
        img = labels["img"]
        if self.hgain or self.sgain or self.vgain:
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
        return labels
    
class BaseMixTransform:
    """
    Class for base mix (MixUp/Mosaic) transformations.

    This implementation is from mmyolo.
    """

    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        """Initializes the BaseMixTransform object with dataset, pre_transform, and probability."""
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = p
    # khai custom 
    def __call__(self, labels):
        """Applies pre-processing transforms and mixup/mosaic transforms to labels data."""
        if random.uniform(0, 1) > self.p:
            return labels

        # Get index of one or three other images
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]

        # Get images information will be used for Mosaic or MixUp
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes] 
        """
        Khi dòng này được thực thi, get_image_and_label được gọi lặp lại cho mỗi index trong indexes, 
        và do đó update_labels_info cũng được gọi lại, xử lý nhãn cho mỗi ảnh sau khi áp dụng augmentations.
        """
        # mix_labels = [self.dataset.get_label_info(i) for i in indexes]
        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels["mix_labels"] = mix_labels

        # Mosaic or MixUp
        labels = self._mix_transform(labels)
        labels.pop("mix_labels", None)
        return labels

    def _mix_transform(self, labels):
        """Applies MixUp or Mosaic augmentation to the label dictionary."""
        raise NotImplementedError

    def get_indexes(self):
        """Gets a list of shuffled indexes for mosaic augmentation."""
        raise NotImplementedError

class Mosaic(BaseMixTransform):
    """
    Mosaic augmentation.

    This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image.
    The augmentation is applied to a dataset with a given probability.

    Attributes:
        dataset: The dataset on which the mosaic augmentation is applied.
        imgsz (int, optional): Image size (height and width) after mosaic pipeline of a single image. Default to 640.
        p (float, optional): Probability of applying the mosaic augmentation. Must be in the range 0-1. Default to 1.0.
        n (int, optional): The grid size, either 4 (for 2x2) or 9 (for 3x3).
    """

    def __init__(self, dataset, imgsz=640, p=1.0, n=4):
        """Initializes the object with a dataset, image size, probability, and border."""
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."
        assert n in (4, 9), "grid must be equal to 4 or 9."
        super().__init__(dataset=dataset, p=p)
        self.dataset = dataset
        self.imgsz = imgsz
        self.border = (-imgsz // 2, -imgsz // 2)  # width, height
        self.n = n

    def get_indexes(self, buffer=True):
        """Return a list of random indexes from the dataset."""
        if buffer:  # select images from buffer
            return random.choices(list(self.dataset.buffer), k=self.n - 1)
        else:  # select any images
            return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

    def _mix_transform(self, labels):
        """Apply mixup transformation to the input image and labels."""
        assert labels.get("rect_shape", None) is None, "rect and mosaic are mutually exclusive."
        assert len(labels.get("mix_labels", [])), "There are no other images for mosaic augment."
        return (
            self._mosaic3(labels) if self.n == 3 else self._mosaic4(labels) if self.n == 4 else self._mosaic9(labels)
        )  # This code is modified for mosaic3 method.

    def _mosaic3(self, labels):
        """Create a 1x3 image mosaic."""
        mosaic_labels = []
        s = self.imgsz
        for i in range(3):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # Load image
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            # Place img in img3
            if i == 0:  # center
                img3 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 3 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 2:  # left
                c = s - w, s + h0 - h, s, s + h0

            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            img3[y1:y2, x1:x2] = img[y1 - padh :, x1 - padw :]  # img3[ymin:ymax, xmin:xmax]
            # hp, wp = h, w  # height, width previous for next iteration

            # Labels assuming imgsz*2 mosaic size
            labels_patch = self._update_labels(labels_patch, padw + self.border[0], padh + self.border[1])
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)

        final_labels["img"] = img3[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]
        return final_labels

    def _mosaic4(self, labels):
        """Create a 2x2 image mosaic."""
        mosaic_labels = []
        s = self.imgsz
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)  # mosaic center x, y
        for i in range(4):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # Load image
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            # Place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            labels_patch = self._update_labels(labels_patch, padw, padh)
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        final_labels["img"] = img4
        return final_labels

    def _mosaic9(self, labels):
        """Create a 3x3 image mosaic."""
        mosaic_labels = []
        s = self.imgsz
        hp, wp = -1, -1  # height, width previous
        for i in range(9):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # Load image
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            # Place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            # Image
            img9[y1:y2, x1:x2] = img[y1 - padh :, x1 - padw :]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous for next iteration

            # Labels assuming imgsz*2 mosaic size
            labels_patch = self._update_labels(labels_patch, padw + self.border[0], padh + self.border[1])
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)

        final_labels["img"] = img9[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]
        return final_labels

    @staticmethod
    def _update_labels(labels, padw, padh):
        """Update labels."""
        nh, nw = labels["img"].shape[:2]
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(nw, nh)
        labels["instances"].add_padding(padw, padh)
        return labels
    
    #khai custom 
    def _cat_labels(self, mosaic_labels):
        """Return labels with mosaic border instances clipped."""
        if len(mosaic_labels) == 0:
            return {}
        cls_obj = []
        cls_color = []
        instances = []
        imgsz = self.imgsz * 2  # mosaic imgsz
        for labels in mosaic_labels:
            cls_obj.append(labels["cls_obj"])
            cls_color.append(labels["cls_color"])
            instances.append(labels["instances"])
        # Final labels
        final_labels = {
            "im_file": mosaic_labels[0]["im_file"],
            "ori_shape": mosaic_labels[0]["ori_shape"],
            "resized_shape": (imgsz, imgsz),
            "cls_obj": np.concatenate(cls_obj, 0),
            "cls_color": np.concatenate(cls_color, 0),
            "instances": Instances.concatenate(instances, axis=0),
            "mosaic_border": self.border,
        }
        final_labels["instances"].clip(imgsz, imgsz)
        good = final_labels["instances"].remove_zero_area_boxes()
        final_labels["cls_obj"] = final_labels["cls_obj"][good]
        final_labels["cls_color"] = final_labels["cls_color"][good]
        return final_labels

class RandomPerspective:
    """
    Implements random perspective and affine transformations on images and corresponding annotations.

    This class applies random rotations, translations, scaling, shearing, and perspective transformations
    to images and their associated bounding boxes, segments, and keypoints. It can be used as part of an
    augmentation pipeline for object detection and instance segmentation tasks.

    Attributes:
        degrees (float): Maximum absolute degree range for random rotations.
        translate (float): Maximum translation as a fraction of the image size.
        scale (float): Scaling factor range, e.g., scale=0.1 means 0.9-1.1.
        shear (float): Maximum shear angle in degrees.
        perspective (float): Perspective distortion factor.
        border (Tuple[int, int]): Mosaic border size as (x, y).
        pre_transform (Callable | None): Optional transform to apply before the random perspective.

    Methods:
        affine_transform: Applies affine transformations to the input image.
        apply_bboxes: Transforms bounding boxes using the affine matrix.
        apply_segments: Transforms segments and generates new bounding boxes.
        apply_keypoints: Transforms keypoints using the affine matrix.
        __call__: Applies the random perspective transformation to images and annotations.
        box_candidates: Filters transformed bounding boxes based on size and aspect ratio.

    Examples:
        >>> transform = RandomPerspective(degrees=10, translate=0.1, scale=0.1, shear=10)
        >>> image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        >>> labels = {'img': image, 'cls': np.array([0, 1]), 'instances': Instances(...)}
        >>> result = transform(labels)
        >>> transformed_image = result['img']
        >>> transformed_instances = result['instances']
    """

    def __init__(
        self, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, border=(0, 0), pre_transform=None
    ):
        """
        Initializes RandomPerspective object with transformation parameters.

        This class implements random perspective and affine transformations on images and corresponding bounding boxes,
        segments, and keypoints. Transformations include rotation, translation, scaling, and shearing.

        Args:
            degrees (float): Degree range for random rotations.
            translate (float): Fraction of total width and height for random translation.
            scale (float): Scaling factor interval, e.g., a scale factor of 0.5 allows a resize between 50%-150%.
            shear (float): Shear intensity (angle in degrees).
            perspective (float): Perspective distortion factor.
            border (Tuple[int, int]): Tuple specifying mosaic border (top/bottom, left/right).
            pre_transform (Callable | None): Function/transform to apply to the image before starting the random
                transformation.

        Examples:
            >>> transform = RandomPerspective(degrees=10.0, translate=0.1, scale=0.5, shear=5.0)
            >>> result = transform(labels)  # Apply random perspective to labels
        """

        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border  # mosaic border
        self.pre_transform = pre_transform

    def affine_transform(self, img, border):
        """
        Applies a sequence of affine transformations centered around the image center.

        This function performs a series of geometric transformations on the input image, including
        translation, perspective change, rotation, scaling, and shearing. The transformations are
        applied in a specific order to maintain consistency.

        Args:
            img (np.ndarray): Input image to be transformed.
            border (Tuple[int, int]): Border dimensions for the transformed image.

        Returns:
            (Tuple[np.ndarray, np.ndarray, float]): A tuple containing:
                - np.ndarray: Transformed image.
                - np.ndarray: 3x3 transformation matrix.
                - float: Scale factor applied during the transformation.

        Examples:
            >>> import numpy as np
            >>> img = np.random.rand(100, 100, 3)
            >>> border = (10, 10)
            >>> transformed_img, matrix, scale = affine_transform(img, border)
        """

        # Center
        C = np.eye(3, dtype=np.float32)

        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - self.scale, 1 + self.scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0]  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1]  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        # Affine image
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=(114, 114, 114))
        return img, M, s

    def apply_bboxes(self, bboxes, M):
        """
        Apply affine transformation to bounding boxes.

        This function applies an affine transformation to a set of bounding boxes using the provided
        transformation matrix.

        Args:
            bboxes (torch.Tensor): Bounding boxes in xyxy format with shape (N, 4), where N is the number
                of bounding boxes.
            M (torch.Tensor): Affine transformation matrix with shape (3, 3).

        Returns:
            (torch.Tensor): Transformed bounding boxes in xyxy format with shape (N, 4).

        Examples:
            >>> bboxes = torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40]])
            >>> M = torch.eye(3)
            >>> transformed_bboxes = apply_bboxes(bboxes, M)
        """
        n = len(bboxes)
        if n == 0:
            return bboxes

        xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

        # Create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        return np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype).reshape(4, n).T


    def __call__(self, labels):
        """
        Applies random perspective and affine transformations to an image and its associated labels.

        This method performs a series of transformations including rotation, translation, scaling, shearing,
        and perspective distortion on the input image and adjusts the corresponding bounding boxes, segments,
        and keypoints accordingly.

        Args:
            labels (Dict): A dictionary containing image data and annotations.
                Must include:
                    'img' (ndarray): The input image.
                    'cls' (ndarray): Class labels.
                    'instances' (Instances): Object instances with bounding boxes, segments, and keypoints.
                May include:
                    'mosaic_border' (Tuple[int, int]): Border size for mosaic augmentation.

        Returns:
            (Dict): Transformed labels dictionary containing:
                - 'img' (np.ndarray): The transformed image.
                - 'cls' (np.ndarray): Updated class labels.
                - 'instances' (Instances): Updated object instances.
                - 'resized_shape' (Tuple[int, int]): New image shape after transformation.

        Examples:
            >>> transform = RandomPerspective()
            >>> image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            >>> labels = {
            ...     'img': image,
            ...     'cls': np.array([0, 1, 2]),
            ...     'instances': Instances(bboxes=np.array([[10, 10, 50, 50], [100, 100, 150, 150]]))
            ... }
            >>> result = transform(labels)
            >>> assert result['img'].shape[:2] == result['resized_shape']
        """
        if self.pre_transform and "mosaic_border" not in labels:
            labels = self.pre_transform(labels)
        labels.pop("ratio_pad", None)  # do not need ratio pad

        img = labels["img"]
        cls_obj = labels["cls_obj"]
        cls_color = labels["cls_color"]
        instances = labels.pop("instances")
        # Make sure the coord formats are right
        instances.convert_bbox(format="xyxy")
        instances.denormalize(*img.shape[:2][::-1])

        border = labels.pop("mosaic_border", self.border)
        self.size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2  # w, h
        # M is affine matrix
        # Scale for func:`box_candidates`
        img, M, scale = self.affine_transform(img, border)

        bboxes = self.apply_bboxes(instances.bboxes, M)

        new_instances = Instances(bboxes, bbox_format="xyxy", normalized=False)
        # Clip
        new_instances.clip(*self.size)

        # Filter instances
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        # Make the bboxes have the same scale with new_bboxes
        i = self.box_candidates(
            box1=instances.bboxes.T, box2=new_instances.bboxes.T, area_thr=0.01
        )
        labels["instances"] = new_instances[i]
        labels["cls_obj"] = cls_obj[i]
        labels["cls_color"] = cls_color[i]
        labels["img"] = img
        labels["resized_shape"] = img.shape[:2]
        return labels

    def box_candidates(self, box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
        """
        Compute candidate boxes for further processing based on size and aspect ratio criteria.

        This method compares boxes before and after augmentation to determine if they meet specified
        thresholds for width, height, aspect ratio, and area. It's used to filter out boxes that have
        been overly distorted or reduced by the augmentation process.

        Args:
            box1 (numpy.ndarray): Original boxes before augmentation, shape (4, N) where n is the
                number of boxes. Format is [x1, y1, x2, y2] in absolute coordinates.
            box2 (numpy.ndarray): Augmented boxes after transformation, shape (4, N). Format is
                [x1, y1, x2, y2] in absolute coordinates.
            wh_thr (float): Width and height threshold in pixels. Boxes smaller than this in either
                dimension are rejected.
            ar_thr (float): Aspect ratio threshold. Boxes with an aspect ratio greater than this
                value are rejected.
            area_thr (float): Area ratio threshold. Boxes with an area ratio (new/old) less than
                this value are rejected.
            eps (float): Small epsilon value to prevent division by zero.

        Returns:
            (numpy.ndarray): Boolean array of shape (n,) indicating which boxes are candidates.
                True values correspond to boxes that meet all criteria.

        Examples:
            >>> random_perspective = RandomPerspective()
            >>> box1 = np.array([[0, 0, 100, 100], [0, 0, 50, 50]]).T
            >>> box2 = np.array([[10, 10, 90, 90], [5, 5, 45, 45]]).T
            >>> candidates = random_perspective.box_candidates(box1, box2)
            >>> print(candidates)
            [True True]
        """
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates

#khai custom 
class Albumentations:
    """
    Albumentations transformations for image augmentation.

    This class applies various image transformations using the Albumentations library. It includes operations such as
    Blur, Median Blur, conversion to grayscale, Contrast Limited Adaptive Histogram Equalization (CLAHE), random changes
    in brightness and contrast, RandomGamma, and image quality reduction through compression.

    Attributes:
        p (float): Probability of applying the transformations.
        transform (albumentations.Compose): Composed Albumentations transforms.
        contains_spatial (bool): Indicates if the transforms include spatial operations.

    Methods:
        __call__: Applies the Albumentations transformations to the input labels.

    Examples:
        >>> transform = Albumentations(p=0.5)
        >>> augmented_labels = transform(labels)

    Notes:
        - The Albumentations package must be installed to use this class.
        - If the package is not installed or an error occurs during initialization, the transform will be set to None.
        - Spatial transforms are handled differently and require special processing for bounding boxes.
    """

    def __init__(self, p=1.0):
        """
        Initialize the Albumentations transform object for YOLO bbox formatted parameters.

        This class applies various image augmentations using the Albumentations library, including Blur, Median Blur,
        conversion to grayscale, Contrast Limited Adaptive Histogram Equalization, random changes of brightness and
        contrast, RandomGamma, and image quality reduction through compression.

        Args:
            p (float): Probability of applying the augmentations. Must be between 0 and 1.

        Attributes:
            p (float): Probability of applying the augmentations.
            transform (albumentations.Compose): Composed Albumentations transforms.
            contains_spatial (bool): Indicates if the transforms include spatial transformations.

        Raises:
            ImportError: If the Albumentations package is not installed.
            Exception: For any other errors during initialization.

        Examples:
            >>> transform = Albumentations(p=0.5)
            >>> augmented = transform(image=image, bboxes=bboxes, class_labels=classes)
            >>> augmented_image = augmented['image']
            >>> augmented_bboxes = augmented['bboxes']

        Notes:
            - Requires Albumentations version 1.0.3 or higher.
            - Spatial transforms are handled differently to ensure bbox compatibility.
            - Some transforms are applied with very low probability (0.01) by default.
        """
        self.p = p
        self.transform = None
        prefix = colorstr("albumentations: ")

        try:
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)  # version requirement

            # List of possible spatial transforms
            spatial_transforms = {
                "Affine",
                "BBoxSafeRandomCrop",
                "CenterCrop",
                "CoarseDropout",
                "Crop",
                "CropAndPad",
                "CropNonEmptyMaskIfExists",
                "D4",
                "ElasticTransform",
                "Flip",
                "GridDistortion",
                "GridDropout",
                "HorizontalFlip",
                "Lambda",
                "LongestMaxSize",
                "MaskDropout",
                "MixUp",
                "Morphological",
                "NoOp",
                "OpticalDistortion",
                "PadIfNeeded",
                "Perspective",
                "PiecewiseAffine",
                "PixelDropout",
                "RandomCrop",
                "RandomCropFromBorders",
                "RandomGridShuffle",
                "RandomResizedCrop",
                "RandomRotate90",
                "RandomScale",
                "RandomSizedBBoxSafeCrop",
                "RandomSizedCrop",
                "Resize",
                "Rotate",
                "SafeRotate",
                "ShiftScaleRotate",
                "SmallestMaxSize",
                "Transpose",
                "VerticalFlip",
                "XYMasking",
            }  # from https://albumentations.ai/docs/getting_started/transforms_and_targets/#spatial-level-transforms

            # Transforms
            T = [
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0),
            ]

            # Compose transforms
            self.contains_spatial = any(transform.__class__.__name__ in spatial_transforms for transform in T)
            self.transform = (
                A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
                if self.contains_spatial
                else A.Compose(T)
            )
            LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

    def __call__(self, labels):
        """
        Applies Albumentations transformations to input labels.

        This method applies a series of image augmentations using the Albumentations library. It can perform both
        spatial and non-spatial transformations on the input image and its corresponding labels.

        Args:
            labels (Dict): A dictionary containing image data and annotations. Expected keys are:
                - 'img': numpy.ndarray representing the image
                - 'cls': numpy.ndarray of class labels
                - 'instances': object containing bounding boxes and other instance information

        Returns:
            (Dict): The input dictionary with augmented image and updated annotations.

        Examples:
            >>> transform = Albumentations(p=0.5)
            >>> labels = {
            ...     "img": np.random.rand(640, 640, 3),
            ...     "cls": np.array([0, 1]),
            ...     "instances": Instances(bboxes=np.array([[0, 0, 1, 1], [0.5, 0.5, 0.8, 0.8]]))
            ... }
            >>> augmented = transform(labels)
            >>> assert augmented["img"].shape == (640, 640, 3)

        Notes:
            - The method applies transformations with probability self.p.
            - Spatial transforms update bounding boxes, while non-spatial transforms only modify the image.
            - Requires the Albumentations library to be installed.
        """
        if self.transform is None or random.random() > self.p:
            return labels

        if self.contains_spatial:
            cls_obj = labels["cls_obj"]
            cls_color = labels["cls_color"]
            # print(f"cls_augment:{cls}")
            if len(cls_obj)>0:
                instances.convert_bbox("xywh")
                instances.normalize(*im.shape[:2][::-1])
                bboxes = instances.bboxes
                
                new = self.transform(image = im, bboxes = bboxes, class_label = cls_obj)
                
            # if len(cls):
            #     im = labels["img"]
            #     labels["instances"].convert_bbox("xywh")
            #     labels["instances"].normalize(*im.shape[:2][::-1])
            #     bboxes = labels["instances"].bboxes
            #     # TODO: add supports of segments and keypoints
            #     new = self.transform(image=im, bboxes=bboxes, class_labels=cls)  # transformed
                
            #     print("Original class_labels_augment:", cls)
            #     print("Transformed class_labels_augment:", new["class_labels"])
                
                if len(new["class_labels"]) > 0:  # skip update if no bbox in new im
                    labels["img"] = new["image"]
                    labels["cls_obj"] = np.array(new["class_labels"])
                    bboxes = np.array(new["bboxes"], dtype=np.float32)
                    labels["cls_color"] = cls_color
                labels["instances"].update(bboxes=bboxes)
        else:
            labels["img"] = self.transform(image=labels["img"])["image"]  # transformed

        return labels
    
class MixUp(BaseMixTransform):
    """Class for applying MixUp augmentation to the dataset."""

    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        """Initializes MixUp object with dataset, pre_transform, and probability of applying MixUp."""
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)

    def get_indexes(self):
        """Get a random index from the dataset."""
        return random.randint(0, len(self.dataset) - 1)

    def _mix_transform(self, labels):
        """Applies MixUp augmentation as per https://arxiv.org/pdf/1710.09412.pdf."""
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        labels2 = labels["mix_labels"][0]
        labels["img"] = (labels["img"] * r + labels2["img"] * (1 - r)).astype(np.uint8)
        labels["instances"] = Instances.concatenate([labels["instances"], labels2["instances"]], axis=0)
        labels["cls_obj"] = np.concatenate([labels["cls_obj"], labels2["cls_obj"]], 0)
        return labels
    
def v8_transforms(dataset, imgsz, hyp):
    """Convert images to a size suitable for YOLOv8 training."""
    print(type(dataset.data))

    pre_transform = Compose([
        Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic),
        CopyPaste(p=hyp.copy_paste),
        RandomPerspective(
            degrees=hyp.degrees,
            translate=hyp.translate,
            scale=hyp.scale,
            shear=hyp.shear,
            perspective=hyp.perspective,
            pre_transform=LetterBox(new_shape=(imgsz, imgsz)),
        )])
    if isinstance(dataset.data, dict):
        flip_idx = dataset.data.get('flip_idx', None)  # for keypoints augmentation
    else:
        flip_idx = None
        LOGGER.warning("WARNING ⚠️ dataset.data is not a dictionary.")
    
    # if dataset.use_keypoints and flip_idx is None and hyp.fliplr > 0.0:
    #     hyp.fliplr = 0.0
    #     LOGGER.warning("WARNING ⚠️ No `flip_idx` provided while training keypoints, setting augmentation 'fliplr=0.0'")
    
    return Compose([
        pre_transform,
        MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
        Albumentations(p=1.0),
        RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
        RandomFlip(direction='vertical', p=hyp.flipud),
        RandomFlip(direction='horizontal', p=hyp.fliplr, flip_idx=flip_idx)
    ])  # transforms

# khai custom     
class Format:
    """
    Formats image annotations for object detection, instance segmentation, and pose estimation tasks. The class
    standardizes the image and instance annotations to be used by the `collate_fn` in PyTorch DataLoader.

    Attributes:
        bbox_format (str): Format for bounding boxes. Default is 'xywh'.
        normalize (bool): Whether to normalize bounding boxes. Default is True.
        return_mask (bool): Return instance masks for segmentation. Default is False.
        return_keypoint (bool): Return keypoints for pose estimation. Default is False.
        mask_ratio (int): Downsample ratio for masks. Default is 4.
        mask_overlap (bool): Whether to overlap masks. Default is True.
        batch_idx (bool): Keep batch indexes. Default is True.
    """

    def __init__(
        self,
        bbox_format="xywh",
        normalize=True,
        mask_ratio=4,
        mask_overlap=True,
        batch_idx=True,
    ):
        """Initializes the Format class with given parameters."""
        self.bbox_format = bbox_format
        self.normalize = normalize
        self.mask_ratio = mask_ratio
        self.mask_overlap = mask_overlap
        self.batch_idx = batch_idx  # keep the batch indexes

    # collate_fn called back in here  
    def __call__(self, labels):
        """Return formatted image, classes, bounding boxes & keypoints to be used by 'collate_fn'."""
        img = labels.pop("img")
        h, w = img.shape[:2]
        cls_obj = labels.pop("cls_obj")
        cls_color = labels.pop("cls_color")
        # print(f"khai_bug_augment:{cls}")
        instances = labels.pop("instances")
        instances.convert_bbox(format=self.bbox_format)
        instances.denormalize(w, h)
        nl = len(instances)
        if self.normalize:
            instances.normalize(w, h)
        labels["img"] = self._format_img(img)
        labels["cls_obj"] = torch.from_numpy(cls_obj) if nl else torch.zeros(nl)
        labels["cls_color"] = torch.from_numpy(cls_color) if nl else torch.zeros(nl)
        labels["bboxes"] = torch.from_numpy(instances.bboxes) if nl else torch.zeros((nl, 4))
        # Then we can use collate_fn
        if self.batch_idx:
            labels["batch_idx"] = torch.zeros(nl)
        return labels

    def _format_img(self, img):
        """Format the image for YOLO from Numpy array to PyTorch tensor."""
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = np.ascontiguousarray(img.transpose(2, 0, 1)[::-1])
        img = torch.from_numpy(img)
        return img
    
class CopyPaste:
    """
    Implements Copy-Paste augmentation as described in https://arxiv.org/abs/2012.07177.

    This class applies Copy-Paste augmentation on images and their corresponding instances.

    Attributes:
        p (float): Probability of applying the Copy-Paste augmentation. Must be between 0 and 1.

    Methods:
        __call__: Applies Copy-Paste augmentation to given image and instances.

    Examples:
        >>> copypaste = CopyPaste(p=0.5)
        >>> augmented_labels = copypaste(labels)
        >>> augmented_image = augmented_labels['img']
    """

    def __init__(self, p=0.5) -> None:
        """
        Initializes the CopyPaste augmentation object.

        This class implements the Copy-Paste augmentation as described in the paper "Simple Copy-Paste is a Strong Data
        Augmentation Method for Instance Segmentation" (https://arxiv.org/abs/2012.07177). It applies the Copy-Paste
        augmentation on images and their corresponding instances with a given probability.

        Args:
            p (float): The probability of applying the Copy-Paste augmentation. Must be between 0 and 1.

        Attributes:
            p (float): Stores the probability of applying the augmentation.

        Examples:
            >>> augment = CopyPaste(p=0.7)
            >>> augmented_data = augment(original_data)
        """
        self.p = p

    def __call__(self, labels):
        # """
        # Applies Copy-Paste augmentation to an image and its instances.

        # Args:
        #     labels (Dict): A dictionary containing:
        #         - 'img' (np.ndarray): The image to augment.
        #         - 'cls' (np.ndarray): Class labels for the instances.
        #         - 'instances' (ultralytics.engine.results.Instances): Object containing bounding boxes, segments, etc.

        # Returns:
        #     (Dict): Dictionary with augmented image and updated instances under 'img', 'cls', and 'instances' keys.

        # Examples:
        #     >>> labels = {'img': np.random.rand(640, 640, 3), 'cls': np.array([0, 1, 2]), 'instances': Instances(...)}
        #     >>> augmenter = CopyPaste(p=0.5)
        #     >>> augmented_labels = augmenter(labels)
        # """
        im = labels["img"]
        cls_obj = labels["cls_obj"]
        cls_color = labels["cls_color"]
        h, w = im.shape[:2]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xyxy")
        instances.denormalize(w, h)
        # if self.p and len(instances.bboxes):
        if self.p and len(instances.bboxes):
            n = len(instances)
            _, w, _ = im.shape  # height, width, channels
            im_new = np.zeros(im.shape, np.uint8)

            # Calculate ioa first then select indexes randomly
            ins_flip = deepcopy(instances)
            ins_flip.fliplr(w)

            ioa = bbox_ioa(ins_flip.bboxes, instances.bboxes)  # intersection over area, (N, M)
            indexes = np.nonzero((ioa < 0.30).all(1))[0]  # (N, )
            n = len(indexes)
            for j in random.sample(list(indexes), k=round(self.p * n)):
                cls_obj = np.concatenate((cls_obj, cls_obj[[j]]), axis=0)
                cls_color = np.concatenate((cls_color, cls_color[[j]]), axis=0)
                instances = Instances.concatenate((instances, ins_flip[[j]]), axis=0)
                # cv2.drawContours(im_new, instances.segments[[j]].astype(np.int32), -1, (1, 1, 1), cv2.FILLED)

            result = cv2.flip(im, 1)  # augment segments (flip left-right)
            i = cv2.flip(im_new, 1).astype(bool)
            im[i] = result[i]

        labels["img"] = im
        labels["cls_obj"] = cls_obj
        labels["cls_color"] = cls_color 
        labels["instances"] = instances
        return labels
    
def classify_transforms(
    size=224,
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
    interpolation="BILINEAR",
    crop_fraction: float = DEFAULT_CROP_FRACTION,
):
    """
    Creates a composition of image transforms for classification tasks.

    This function generates a sequence of torchvision transforms suitable for preprocessing images
    for classification models during evaluation or inference. The transforms include resizing,
    center cropping, conversion to tensor, and normalization.

    Args:
        size (int | tuple): The target size for the transformed image. If an int, it defines the shortest edge. If a
            tuple, it defines (height, width).
        mean (tuple): Mean values for each RGB channel used in normalization.
        std (tuple): Standard deviation values for each RGB channel used in normalization.
        interpolation (str): Interpolation method of either 'NEAREST', 'BILINEAR' or 'BICUBIC'.
        crop_fraction (float): Fraction of the image to be cropped.

    Returns:
        (torchvision.transforms.Compose): A composition of torchvision transforms.

    Examples:
        >>> transforms = classify_transforms(size=224)
        >>> img = Image.open('path/to/image.jpg')
        >>> transformed_img = transforms(img)
    """
    import torchvision.transforms as T  # scope for faster 'import ultralytics'

    if isinstance(size, (tuple, list)):
        assert len(size) == 2, f"'size' tuples must be length 2, not length {len(size)}"
        scale_size = tuple(math.floor(x / crop_fraction) for x in size)
    else:
        scale_size = math.floor(size / crop_fraction)
        scale_size = (scale_size, scale_size)

    # Aspect ratio is preserved, crops center within image, no borders are added, image is lost
    if scale_size[0] == scale_size[1]:
        # Simple case, use torchvision built-in Resize with the shortest edge mode (scalar size arg)
        tfl = [T.Resize(scale_size[0], interpolation=getattr(T.InterpolationMode, interpolation))]
    else:
        # Resize the shortest edge to matching target dim for non-square target
        tfl = [T.Resize(scale_size)]
    tfl.extend(
        [
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]
    )
    return T.Compose(tfl)

class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose."""

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
        """Initialize LetterBox object with specific parameters."""
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left

    def __call__(self, labels=None, image=None):
        """Return updated labels and image with added border."""
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img
    #khai custom 
    def _update_labels(self, labels, ratio, padw, padh):
        print(f"Labels trước khi cập nhật: {labels}")
        if "instances" in labels:
            print(f"Instances trước khi denormalize: {labels['instances']}")
        labels["instances"].denormalize(nw, nh)
        """Update labels."""
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels
