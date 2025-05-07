import os
import numpy as np
from skimage.measure import label, regionprops
import cv2
import skimage.io as sio

import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, root_dir, class_to_idx, transforms=None, train=True):
        self.root_dir = root_dir
        self.transforms = transforms
        self.train = train
        self.class_to_idx = class_to_idx

        if train:
            self.img_dirs = [
                os.path.join(root_dir, d) for d in os.listdir(root_dir)
            ]
        else:
            self.img_files = [
                os.path.join(root_dir, f)
                for f in os.listdir(root_dir)
                if f.endswith('.tif')
            ]

    def __len__(self):
        return len(self.img_dirs) if self.train else len(self.img_files)

    def __getitem__(self, idx):
        if self.train:
            img_dir = self.img_dirs[idx]
            img_path = os.path.join(img_dir, 'image.tif')

            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Cannot read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Get all class masks
            masks = []
            labels = []
            for mask_file in os.listdir(img_dir):
                if mask_file != 'image.tif':
                    class_name = mask_file.split('.')[0]
                    mask_path = os.path.join(img_dir, mask_file)
                    mask = sio.imread(mask_path)
                    if mask is None:
                        raise ValueError(f"Cannot read mask: {mask_path}")

                    # Handle binary or multi-instance masks
                    if mask.max() == 1:  # Binary mask
                        masks.append(mask)
                        labels.append(self.class_to_idx[class_name])
                    else:  # Multi-instance mask, need to separate
                        labeled_mask = label(mask)
                        for region in regionprops(labeled_mask):
                            instance_mask = np.zeros_like(mask)
                            instance_mask[labeled_mask == region.label] = 1
                            masks.append(instance_mask)
                            labels.append(self.class_to_idx[class_name])

            # Handle rare case where no masks found
            if len(masks) == 0:
                # Create dummy mask and label
                masks = [
                    np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                ]
                labels = [0]  # Background class

            # Apply consistent augmentations to image and masks
            if self.transforms is not None:
                transformed = self.transforms(image=image, masks=masks)
                image = transformed['image']
                masks = transformed['masks']

            # Convert to proper format for Mask R-CNN
            # Masks need to be binary (0 or 1)
            masks = [np.asarray(m, dtype=np.uint8) for m in masks]

            # Compute bounding boxes from masks
            boxes = []
            valid_masks = []
            valid_labels = []

            for mask, label_idx in zip(masks, labels):
                pos = np.where(mask > 0)
                if len(pos[0]) > 0:  # Only add box if mask has pixels
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])

                    # Ensure box has area and valid dimensions (add padding)
                    if xmax <= xmin:
                        delta = 1
                        xmax = xmin + delta
                    if ymax <= ymin:
                        delta = 1
                        ymax = ymin + delta

                    # Store valid mask, box and label
                    valid_masks.append(mask)
                    boxes.append([xmin, ymin, xmax, ymax])
                    valid_labels.append(label_idx)

            # In case all masks were filtered out
            if len(valid_masks) == 0:
                valid_masks = [
                    np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                ]
                boxes = [[0, 0, 1, 1]]
                valid_labels = [0]  # Background class

            # Convert to tensors
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(valid_labels, dtype=torch.int64)

            # Resize with correct shape for Mask R-CNN [N, H, W]
            masks_np = np.array(valid_masks, dtype=np.uint8)
            masks = torch.as_tensor(masks_np, dtype=torch.uint8)

            # Create target dictionary
            target = {}
            target['boxes'] = boxes
            target['labels'] = labels
            target['masks'] = masks

            return image, target

        else:  # Test mode
            img_path = self.img_files[idx]
            image_id = os.path.basename(img_path).split('.')[0]

            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Cannot read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Apply transformations
            if self.transforms is not None:
                transformed = self.transforms(image=image)
                image = transformed['image']

            return image, image_id
