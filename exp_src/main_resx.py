import os
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
import json
import matplotlib.pyplot as plt
import cv2
import random
from tqdm import tqdm
import skimage.io as sio
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import zipfile

import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNPredictor
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils import parse_args, encode_mask

# Configuration
CONFIG = {
    'MODEL_NAME': 'maskrcnn_resx_rerun_2',
    'TRAIN_DATA_PATH': 'data/train',
    'TEST_DATA_PATH': 'data/test_release',
    'CKPT_PATH': 'ckpt',
    'OUTPUT_PATH': 'results',
    'BATCH_SIZE': 2,
    'NUM_EPOCHS': 50,
    'LEARNING_RATE': 5e-5,
    'NUM_WORKERS': 2,
    'CLASS_MAP': {
        'class1': 1,
        'class2': 2,
        'class3': 3,
        'class4': 4
    }
}

args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

os.makedirs(CONFIG['OUTPUT_PATH'], exist_ok=True)
os.makedirs(CONFIG['CKPT_PATH'], exist_ok=True)

# Create class name to idx and idx to class name mapping
CLASS_TO_IDX = CONFIG['CLASS_MAP']
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

# Image filename to idx mapping
IMAGE_NAME_TO_IDX = {}
try:
    with open("data/test_image_name_to_ids.json", "r") as f:
        image_data = json.load(f)
        for item in image_data:
            file_name = item.get("file_name", "")
            image_id = item.get("id", 0)
            if file_name:
                IMAGE_NAME_TO_IDX[file_name] = image_id
                base_name = file_name.split('.')[0]
                IMAGE_NAME_TO_IDX[base_name] = image_id
except Exception as e:
    print(f"Warning: Could not load image name to ID mappings: {e}")
    IMAGE_NAME_TO_IDX = {}


class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transforms=None, train=True):
        self.root_dir = root_dir
        self.transforms = transforms
        self.train = train

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
                        labels.append(CLASS_TO_IDX[class_name])
                    else:  # Multi-instance mask, need to separate
                        labeled_mask = label(mask)
                        for region in regionprops(labeled_mask):
                            instance_mask = np.zeros_like(mask)
                            instance_mask[labeled_mask == region.label] = 1
                            masks.append(instance_mask)
                            labels.append(CLASS_TO_IDX[class_name])

            # Handle rare case where no masks found
            if len(masks) == 0:
                # Create dummy mask and label
                masks = [
                    np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                ]
                labels = [0]  # Background class

            # Apply consistent augmentations to image and masks
            if self.transforms is not None:
                pil_image = Image.fromarray(image)
                pil_masks = [Image.fromarray(m) for m in masks]
                if random.random() < 0.5:
                    pil_image = F.hflip(pil_image)
                    pil_masks = [F.hflip(m) for m in pil_masks]
                if random.random() < 0.5:
                    pil_image = F.vflip(pil_image)
                    pil_masks = [F.vflip(m) for m in pil_masks]
                # Apply color jitter only on the image
                cj = T.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                )
                pil_image = cj(pil_image)
                # Image → Tensor + Normalize
                image = T.ToTensor()(pil_image)
                image = T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )(image)
                # Transform masks to numpy arrays
                masks = [np.asarray(m, dtype=np.uint8) for m in pil_masks]

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
                pil_image = Image.fromarray(image)
                image = self.transforms(pil_image)

            return image, image_id


class MaskRCNNSegmentation:
    def __init__(self, config=None):
        """
        Initialize the MaskRCNN segmentation model and related components

        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config or CONFIG
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Running on device: {self.device}")
        self.class_to_idx = CLASS_TO_IDX
        self.idx_to_class = IDX_TO_CLASS
        self.num_classes = len(self.class_to_idx) + 1  # +1 for bg class

        # Create output directory
        os.makedirs(self.config['OUTPUT_PATH'], exist_ok=True)
        os.makedirs(self.config['CKPT_PATH'], exist_ok=True)

        # Initialize model
        self.model = self._get_model()
        self.model.to(self.device)

        # Initialize optimizer
        trainable_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        self.optimizer = optim.AdamW(
            trainable_params, lr=self.config['LEARNING_RATE'],
        )
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config['NUM_EPOCHS']
        )

    def _get_model(self):
        """Create and configure the Mask R-CNN model"""
        weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V2
        resnext = resnext50_32x4d(weights=weights)

        # Use ResNeXt backbone with FPN
        return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        in_channels_list = [
            resnext.layer1[-1].conv3.out_channels,
            resnext.layer2[-1].conv3.out_channels,
            resnext.layer3[-1].conv3.out_channels,
            resnext.layer4[-1].conv3.out_channels
        ]

        backbone = BackboneWithFPN(
            resnext,
            return_layers=return_layers,
            in_channels_list=in_channels_list,
            out_channels=256,
        )

        # Freeze backbone layers
        for param in backbone.parameters():
            param.requires_grad = False

        # Load pre-trained model
        model = MaskRCNN(backbone=backbone, num_classes=self.num_classes)

        # Ensure the model is in the training mode
        model.train()

        # print model size
        print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

        return model

    def _get_transforms(self, train=True):
        """Get transformations for data augmentation"""
        if train:
            transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        return transform

    def _collate_fn(self, batch):
        """
        Custom collate function to handle variable-sized data in the batch

        Args:
            batch: List of tuples (image, target)

        Returns:
            Batched images and targets
        """
        # Filter out any problematic samples (e.g., empty targets)
        filtered_batch = []
        for image, target in batch:
            # Skip samples with no valid boxes or invalid dimensions
            if (len(target['boxes']) > 0 and
                all(
                    box[2] > box[0] and box[3] > box[1]
                    for box in target['boxes'])):
                filtered_batch.append((image, target))

        # If all samples were filtered out, create a valid dummy sample
        if len(filtered_batch) == 0:
            # Create a dummy sample with a small valid box
            dummy_image = torch.zeros((3, 512, 512), dtype=torch.float32)
            dummy_target = {
                'boxes': torch.tensor([[10, 10, 20, 20]], dtype=torch.float32),
                'labels': torch.tensor([0], dtype=torch.int64),  # bg class
                'masks': torch.zeros((1, 512, 512), dtype=torch.uint8)
            }
            filtered_batch = [(dummy_image, dummy_target)]

        return tuple(zip(*filtered_batch))

    def prepare_data(self):
        """Prepare datasets and dataloaders with train/val split"""
        # Create datasets
        full_train_dataset = SegmentationDataset(
            root_dir=self.config['TRAIN_DATA_PATH'],
            transforms=self._get_transforms(train=True),
            train=True
        )

        # Create train/val split (80% train, 20% validation)
        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset, [0.8, 0.2]
        )

        self.test_dataset = SegmentationDataset(
            root_dir=self.config['TEST_DATA_PATH'],
            transforms=self._get_transforms(train=False),
            train=False
        )

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['BATCH_SIZE'],
            shuffle=True,
            num_workers=self.config['NUM_WORKERS'],
            collate_fn=self._collate_fn
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['BATCH_SIZE'],
            shuffle=False,
            num_workers=self.config['NUM_WORKERS'],
            collate_fn=self._collate_fn
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config['NUM_WORKERS'],
            collate_fn=lambda x: tuple(zip(*x))
        )

        print(
            f"Prepared {len(self.train_dataset)} training samples, "
            f"{len(self.val_dataset)} validation samples and "
            f"{len(self.test_dataset)} test samples"
        )

    def train(self):
        """Train the model with validation after each epoch"""
        print("Starting training...")
        print(f'Model name: {self.config["MODEL_NAME"]}')

        # Initialize TensorBoard writer
        tb_log_dir = os.path.join('runs', self.config['MODEL_NAME'])
        self.writer = SummaryWriter(tb_log_dir)
        print(f"TensorBoard logs will be saved to {tb_log_dir}")

        best_map = 0.0
        best_ap50 = 0.0

        for epoch in range(self.config['NUM_EPOCHS']):
            # Training phase
            self.model.train()
            running_loss = 0.0

            for i, (images, targets) in enumerate(tqdm(self.train_loader)):
                # Move data to GPU
                images = [image.to(self.device) for image in images]
                targets = [
                    {
                        k: v.to(self.device)
                        for k, v in t.items()
                    }
                    for t in targets
                ]

                try:
                    # Forward pass
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    running_loss += losses.item()

                    # Backward pass
                    self.optimizer.zero_grad()
                    losses.backward()
                    self.optimizer.step()

                    # Log losses to TensorBoard
                    global_step = epoch * len(self.train_loader) + i
                    self.writer.add_scalar(
                        'Train/Training Loss',
                        losses.item(),
                        global_step
                    )
                    self.writer.add_scalar(
                        'Train/Learning Rate',
                        self.optimizer.param_groups[0]['lr'],
                        global_step
                    )
                    for loss_name, loss_value in loss_dict.items():
                        self.writer.add_scalar(
                            f'Train/{loss_name}',
                            loss_value.item(),
                            global_step
                        )
                except RuntimeError as e:
                    print(f"Skipping batch due to error: {e}")
                    continue

            self.lr_scheduler.step()

            # Calculate average losses
            num_batches = len(self.train_loader)
            epoch_loss = running_loss / num_batches if num_batches > 0 else 0

            # Print epoch stats
            print(
                f"Epoch {epoch+1}/{self.config['NUM_EPOCHS']}, "
                f"Loss: {epoch_loss:.4f}"
            )

            # Evaluation phase
            print("Evaluating on validation set...")
            eval_metrics = self.evaluate(self.val_loader)

            # Log evaluation metrics to TensorBoard
            self.writer.add_scalar('Val/mAP', eval_metrics["mAP"], epoch)
            self.writer.add_scalar('Val/AP50', eval_metrics["AP50"], epoch)
            self.writer.add_scalar('Val/AP75', eval_metrics["AP75"], epoch)

            # Print evaluation metrics
            print(f"Validation metrics: mAP: {eval_metrics['mAP']:.4f}, "
                  f"AP50: {eval_metrics['AP50']:.4f}, "
                  f"AP75: {eval_metrics['AP75']:.4f}")

            # Save best model based on mAP
            if eval_metrics["mAP"] > best_map:
                best_map = eval_metrics["mAP"]
                self.save_model(
                    f'{self.config["MODEL_NAME"]}_best.pth',
                    epoch=epoch,
                    map=eval_metrics["mAP"],
                    ap50=eval_metrics["AP50"]
                )
                print(f"New best model saved with mAP: {best_map:.4f}")

            # Save best model based on AP50
            if eval_metrics["AP50"] > best_ap50:
                best_ap50 = eval_metrics["AP50"]
                self.save_model(
                    f'{self.config["MODEL_NAME"]}_best_ap50.pth',
                    epoch=epoch,
                    map=eval_metrics["mAP"],
                    ap50=eval_metrics["AP50"]
                )
                print(f"New best model saved with AP50: {best_ap50:.4f}")

            # Log sample predictions
            if (epoch + 1) % 2 == 0 or epoch == self.config['NUM_EPOCHS'] - 1:
                self._log_sample_predictions(epoch)

            # Save model checkpoint
            if (epoch + 1) % 5 == 0 or epoch == self.config['NUM_EPOCHS'] - 1:
                self.save_model(
                    f'{self.config["MODEL_NAME"]}_epoch_{epoch+1}.pth',
                    epoch=epoch,
                    map=eval_metrics["mAP"],
                    ap50=eval_metrics["AP50"]
                )

        print("Training complete!")
        print(f"Best validation mAP: {best_map:.4f}, AP50: {best_ap50:.4f}")
        self.save_model(f'{self.config["MODEL_NAME"]}_final.pth')

        # Close TensorBoard writer
        self.writer.close()

    # TODO: Maybe a better visuallization
    def _log_sample_predictions(self, epoch):
        """Log sample predictions to TensorBoard"""
        if not hasattr(self, 'train_dataset'):
            return

        # Set model to eval mode
        self.model.eval()

        with torch.no_grad():
            # Get a few sample images
            max_samples = min(10, len(self.train_dataset))
            sample_indices = random.sample(range(max_samples), 3)

            for idx in sample_indices:
                image, target = self.train_dataset[idx]
                image_tensor = image.unsqueeze(0).to(self.device)

                # Get predictions
                predictions = self.model(image_tensor)

                # Convert image tensor for display
                image_np = image.permute(1, 2, 0).cpu().numpy()
                # Denormalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image_np = std * image_np + mean
                image_np = np.clip(image_np, 0, 1)

                # Create a visualization of predictions
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))

                # Original image with ground truth
                ax[0].imshow(image_np)
                ax[0].set_title("Ground Truth")
                for i in range(len(target['boxes'])):
                    box = target['boxes'][i].cpu().numpy()
                    label = target['labels'][i].item()
                    ax[0].add_patch(plt.Rectangle(
                        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                        fill=False, color='green', linewidth=2
                    ))
                    ax[0].text(
                        box[0], box[1],
                        f"{self.idx_to_class.get(label, 'unknown')}",
                        color='white', backgroundcolor='green', fontsize=8
                    )

                # Original image with predictions
                ax[1].imshow(image_np)
                ax[1].set_title("Predictions")

                pred = predictions[0]
                keep = pred['scores'] > 0.5
                boxes = pred['boxes'][keep].cpu().numpy()
                labels = pred['labels'][keep].cpu().numpy()
                scores = pred['scores'][keep].cpu().numpy()

                for i in range(len(boxes)):
                    box = boxes[i]
                    label = labels[i]
                    score = scores[i]
                    ax[1].add_patch(plt.Rectangle(
                        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                        fill=False, color='red', linewidth=2)
                    )
                    ax[1].text(
                        box[0], box[1],
                        f"{self.idx_to_class.get(label, 'unknown')}: "
                        f"{score:.2f}",
                        color='white', backgroundcolor='red', fontsize=8
                    )

                plt.tight_layout()

                # Convert plot to image
                fig.canvas.draw()
                plot_img = np.array(fig.canvas.renderer.buffer_rgba())

                # Close the figure to free memory
                plt.close(fig)

                # Add to tensorboard
                self.writer.add_image(
                    f'predictions/sample_{idx}',
                    plot_img.transpose(2, 0, 1),
                    global_step=epoch
                )

    # TODO: Refactor
    def evaluate(self, data_loader):
        """Evaluate model performance using COCO metrics"""
        self.model.eval()

        # Lists to store predictions and ground truth
        coco_gt = {"images": [], "annotations": [], "categories": []}
        coco_dt = []

        # Add categories
        for class_name, class_id in self.class_to_idx.items():
            coco_gt["categories"].append({
                "id": class_id,
                "name": class_name,
                "supercategory": "none"
            })

        ann_id = 0
        img_id = 0

        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc="Evaluating"):
                # Move data to device
                images = [img.to(self.device) for img in images]

                # Get predictions
                outputs = self.model(images)

                # Process each image in the batch
                for i, (image, target, output) in enumerate(
                    zip(images, targets, outputs)
                ):
                    # Add image info to ground truth
                    h, w = image.shape[-2:]
                    coco_gt["images"].append({
                        "id": img_id,
                        "width": w,
                        "height": h
                    })

                    # Add ground truth annotations
                    boxes_gt = target["boxes"].cpu().numpy()
                    labels_gt = target["labels"].cpu().numpy()
                    masks_gt = target["masks"].cpu().numpy()

                    for j, (box, label_idx, mask) in enumerate(
                        zip(boxes_gt, labels_gt, masks_gt)
                    ):
                        # Convert mask to proper RLE format
                        binary_mask = mask.astype(np.uint8)
                        rle = encode_mask(binary_mask)

                        # Calculate area and bbox
                        area = float((box[2] - box[0]) * (box[3] - box[1]))

                        # Add to ground truth annotations
                        coco_gt["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": int(label_idx),
                            "segmentation": rle,
                            "area": area,
                            "bbox": box.tolist(),
                            "iscrowd": 0
                        })
                        ann_id += 1

                    # Process predictions
                    boxes_pred = output["boxes"].cpu().numpy()
                    labels_pred = output["labels"].cpu().numpy()
                    scores_pred = output["scores"].cpu().numpy()
                    masks_pred = output["masks"].cpu().numpy()

                    # Filter predictions by confidence
                    threshold = 0.5
                    keep = scores_pred > threshold

                    boxes_pred = boxes_pred[keep]
                    labels_pred = labels_pred[keep]
                    scores_pred = scores_pred[keep]
                    masks_pred = masks_pred[keep]

                    for j, (box, label_idx, score, mask) in enumerate(
                        zip(boxes_pred, labels_pred, scores_pred, masks_pred)
                    ):
                        # Convert mask to proper RLE format
                        binary_mask = (mask[0] > 0.5).astype(np.uint8)
                        rle = encode_mask(binary_mask)

                        coco_dt.append({
                            "image_id": img_id,
                            "category_id": int(label_idx),
                            "segmentation": rle,
                            "score": float(score),
                            "bbox": box.tolist()
                        })

                    img_id += 1

        # Create COCO objects for evaluation
        if len(coco_gt["annotations"]) == 0 or len(coco_dt) == 0:
            print("No annotations or detections found for evaluation")
            return {"mAP": 0, "AP50": 0, "AP75": 0}

        # Save temp files for COCO evaluation
        gt_file = os.path.join(
            self.config["OUTPUT_PATH"],
            f'{self.config["MODEL_NAME"]}_temp_gt.json'
        )
        dt_file = os.path.join(
            self.config["OUTPUT_PATH"],
            f'{self.config["MODEL_NAME"]}_temp_dt.json'
        )

        with open(gt_file, "w") as f:
            json.dump(coco_gt, f)

        with open(dt_file, "w") as f:
            json.dump(coco_dt, f)

        # Create COCO objects
        coco_gt = COCO(gt_file)
        coco_dt = coco_gt.loadRes(dt_file)

        # Create COCO evaluator
        coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Clean up temp files
        os.remove(gt_file)
        os.remove(dt_file)

        # Return metrics
        metrics = {
            "mAP": coco_eval.stats[0],  # mAP@IoU=0.50:0.95
            "AP50": coco_eval.stats[1],  # mAP@IoU=0.50
            "AP75": coco_eval.stats[2],  # mAP@IoU=0.75
        }

        return metrics

    def predict(self):
        self.model.eval()

        coco_dt = []

        with torch.no_grad():
            for images, img_names in tqdm(self.test_loader, desc="Testing"):
                images = [img.to(self.device) for img in images]
                outputs = self.model(images)

                # Process each image in the batch
                for i, (image, img_name, output) in enumerate(
                    zip(images, img_names, outputs)
                ):
                    mapped_id = IMAGE_NAME_TO_IDX.get(img_name, 0)
                    if mapped_id == 0:
                        print(f"Warn: Image ID not found for {img_name}")

                    # Process predictions
                    boxes_pred = output["boxes"].cpu().numpy()
                    labels_pred = output["labels"].cpu().numpy()
                    scores_pred = output["scores"].cpu().numpy()
                    masks_pred = output["masks"].cpu().numpy()

                    # Filter predictions by confidence
                    threshold = 0.5
                    keep = scores_pred > threshold

                    boxes_pred = boxes_pred[keep]
                    labels_pred = labels_pred[keep]
                    scores_pred = scores_pred[keep]
                    masks_pred = masks_pred[keep]

                    for j, (box, label_idx, score, mask) in enumerate(
                        zip(boxes_pred, labels_pred, scores_pred, masks_pred)
                    ):
                        # Convert mask to proper RLE format
                        binary_mask = (mask[0] > 0.5).astype(np.uint8)
                        rle = encode_mask(binary_mask)

                        coco_dt.append({
                            "image_id": mapped_id,
                            "bbox": box.tolist(),
                            "score": float(score),
                            "category_id": int(label_idx),
                            "segmentation": rle,
                        })

        return coco_dt

    def test(self):
        results = self.predict()

        # Save results for submission
        json_path = os.path.join(
            self.config["OUTPUT_PATH"], f"{self.config["MODEL_NAME"]}.json"
        )
        zip_path = json_path.replace('.json', '.zip')

        if not os.path.exists(os.path.dirname(zip_path)):
            os.makedirs(os.path.dirname(zip_path), exist_ok=True)

        # Save predictions to JSON file
        with open(json_path, "w") as f:
            json.dump(results, f)
        print(f'Predictions saved to {json_path}')

        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.write(json_path, arcname='test-results.json')
        print(f'ZIP file saved to {zip_path}')

    def save_model(self, filename, epoch=None, map=None, ap50=None):
        """Save model weights to disk"""
        torch.save({
            'epoch': epoch,
            'mAP': map,
            'AP50': ap50,
            'model_state_dict': self.model.state_dict()
        }, os.path.join(self.config['CKPT_PATH'], filename))

        output_path = os.path.join(self.config['CKPT_PATH'], filename)
        print(f"Model saved to {output_path}")

    def load_model(self, model_path):
        """Load model weights from disk"""
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.model.to(self.device)

            print(f'Model loaded from {model_path}')
            print(f'Epoch: {ckpt['epoch']}, '
                  f'mAP: {ckpt['mAP']:.4f}, '
                  f'AP50: {ckpt['AP50']:.4f}')
        else:
            print(f"Model file {model_path} not found")

    def load_best_model(self):
        model_path = os.path.join(
            self.config['CKPT_PATH'],
            f"{self.config['MODEL_NAME']}_best_ap50.pth"
        )
        self.load_model(model_path)

        print(f"Best model loaded from {model_path}")

    def visualize_results(self, results=None, num_samples=5):
        """Visualize sample predictions for qualitative analysis"""
        if results is None:
            # Load results from file if not provided
            results_path = os.path.join(
                self.config['OUTPUT_PATH'], 'results.json'
            )
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = json.load(f)
            else:
                print("No results file found. Run predict() first.")
                return

        visualization_path = os.path.join(self.config['OUTPUT_PATH'],
                                          'visualizations')
        os.makedirs(visualization_path, exist_ok=True)

        sample_indices = random.sample(
            range(len(results)), min(num_samples, len(results))
        )

        plt.figure(figsize=(15, 10))
        for i, idx in enumerate(sample_indices):
            result = results[idx]
            image_id = result['image_id']

            # Find the image in test dataset
            for j in range(len(self.test_dataset)):
                _, img_id = self.test_dataset[j]
                if img_id == image_id:
                    image = self.test_dataset.img_files[j]
                    break

            # Load and process image
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Draw results on image
            for detection in result['detections']:
                category_name = detection['category_name']
                score = detection['score']
                rle = detection['segmentation']

                # Convert RLE to binary mask
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                curr_pos = 0
                for j in range(0, len(rle), 2):
                    curr_pos += rle[j]
                    mask_length = rle[j+1]
                    mask_indices = np.arange(curr_pos, curr_pos + mask_length)
                    mask_indices = mask_indices[mask_indices < mask.size]
                    flat_mask = mask.flatten()
                    flat_mask[mask_indices] = 1
                    mask = flat_mask.reshape(mask.shape)
                    curr_pos += mask_length

                # Apply mask as overlay
                color = np.random.randint(0, 255, 3)
                colored_mask = np.zeros_like(img)
                colored_mask[mask == 1] = color
                img = cv2.addWeighted(img, 1, colored_mask, 0.5, 0)

                # Add text label
                y, x = np.where(mask == 1)
                if len(y) > 0:
                    cv2.putText(
                        img, f"{category_name}: {score:.2f}",
                        (np.min(x), np.min(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                    )

            # Save visualization
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(img)
            plt.title(f"Image: {image_id}")
            plt.axis('off')

            # Save individual image
            cv2.imwrite(
                os.path.join(
                    self.config['OUTPUT_PATH'],
                    'visualizations',
                    f'{image_id}_result.jpg'
                ),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.config['OUTPUT_PATH'],
                'visualizations',
                'sample_results.jpg'
            )
        )
        print(
            f"Visualizations saved to "
            f"{os.path.join(self.config['OUTPUT_PATH'], 'visualizations')}"
        )


def main():
    # Create the segmentation model instance
    segmentation = MaskRCNNSegmentation()

    # Prepare datasets and dataloaders
    segmentation.prepare_data()

    # Train the model
    if args.train:
        segmentation.train()

    # Test the model
    elif args.test:
        if args.ckpt:
            segmentation.load_model(args.ckpt)
        else:
            segmentation.load_best_model()
        segmentation.test()

    # Run inference
    elif args.infer:
        if args.ckpt:
            segmentation.load_model(args.ckpt)
        else:
            segmentation.load_best_model()
        results = segmentation.predict()

        # Visualize results
        segmentation.visualize_results(results)

    else:
        print("Please specify --train, --test, or --infer to run the model.")
        return


if __name__ == "__main__":
    main()
