import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class ClassificationDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images  # List of image arrays
        self.labels = labels  # List of label tensors
        self.transform = transform  # Augmentations

    def threshold_mask(self, mask):
        """Binarizes mask: pixel > 0.5 → 1, else 0"""
        return torch.where(mask > 0.5, torch.tensor(1.0), torch.tensor(0.0))

    def apply_random_cutout(self, image):
        """Applies a small random cutout (1-5% of image size)"""
        c, h, w = image.shape
        cutout_size = random.randint(
            int(0.01 * h * w), int(0.05 * h * w)
        )  # 1-5% area
        cut_h = int(np.sqrt(cutout_size))
        cut_w = cut_h

        x = random.randint(0, w - cut_w)
        y = random.randint(0, h - cut_h)

        image[:, y : y + cut_h, x : x + cut_w] = 0  # Black-out region
        return image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # Still an array
        label = self.labels[idx]  # Already a tensor

        # Convert image to tensor if it's not already
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)

        # Ensure correct shape (C, H, W)
        if len(image.shape) == 2:
            image = image.unsqueeze(0)  # Convert grayscale image to (1, H, W)

        # Apply mask thresholding (only to the 2nd channel)
        image[1] = self.threshold_mask(image[1])

        # Apply random cutout
        # image = self.apply_random_cutout(image)
        gray = image[0].unsqueeze(0)
        mask = image[1].unsqueeze(0)
        stacked_img = torch.cat([gray, mask, gray], dim=0)
        # Apply optional transforms (if provided)
        if self.transform:
            stacked_img = self.transform(stacked_img)
            image[0] = stacked_img[0]
            image[1] = stacked_img[1]

        return image, label  # Label remains a tensor


class SegmentationDataset(torch.utils.data.Dataset):
    """
    Dataset class for segmentation task.
    """

    def __init__(self, image_arr, seg_arr, split="train"):
        self.imgs = image_arr
        self.segs = seg_arr
        self.split = split

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = torch.tensor(self.imgs[idx], dtype=torch.float32) / 255.0
        seg = torch.tensor(self.segs[idx])
        img = img.unsqueeze(0)
        seg = seg.unsqueeze(0)

        image_transform = T.Compose(
            [
                T.Normalize(mean=[0.5], std=[0.5]),
                T.Resize(
                    (512, 512), interpolation=T.InterpolationMode.BILINEAR
                ),
            ]
        )

        mask_transform = T.Compose(
            [
                T.Resize(
                    (512, 512), interpolation=T.InterpolationMode.NEAREST
                )  # Nearest-neighbor for masks
            ]
        )

        train_transform = T.Compose(
            [
                T.RandomErasing(p=0.5, scale=(0.02, 0.1)),
                # T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)
            ]
        )

        img = image_transform(img)
        seg = mask_transform(seg)

        if self.split == "train":
            if random.random() < 0.5:
                img = F.hflip(img)
                seg = F.hflip(seg.to(torch.float32)).to(torch.uint16)
            angle = random.uniform(-10, 10)
            translate = (
                random.uniform(-0.1, 0.1) * img.shape[1],
                random.uniform(-0.1, 0.1) * img.shape[2],
            )
            scale = random.uniform(0.9, 1.1)
            shear = random.uniform(-10, 10)

            img = F.affine(
                img,
                angle,
                translate,
                scale,
                shear,
                interpolation=T.InterpolationMode.BILINEAR,
            )
            seg = F.affine(
                seg,
                angle,
                translate,
                scale,
                shear,
                interpolation=T.InterpolationMode.NEAREST,
            )
            img = train_transform(img)
            # seg = train_transform(seg)

        return img, seg

class RegressionDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images  # List of image arrays
        self.labels = labels  # List of label tensors
        self.transform = transform  # Augmentations
    
    def threshold_mask(self, mask):
        """Binarizes mask: pixel > 0.5 → 1, else 0"""
        return torch.where(mask > 0.5, torch.tensor(1.0), torch.tensor(0.0))
    
    def apply_random_cutout(self, image):
        """Applies a small random cutout (1-5% of image size)"""
        c, h, w = image.shape
        cutout_size = random.randint(int(0.01 * h * w), int(0.05 * h * w))  # 1-5% area
        cut_h = int(np.sqrt(cutout_size))
        cut_w = cut_h

        x = random.randint(0, w - cut_w)
        y = random.randint(0, h - cut_h)

        image[:, y:y+cut_h, x:x+cut_w] = 0  # Black-out region
        return image
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]  # Still an array
        label = self.labels[idx]  # Already a tensor

        # Convert image to tensor if it's not already
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)

        # Ensure correct shape (C, H, W)
        if len(image.shape) == 2:
            image = image.unsqueeze(0)  # Convert grayscale image to (1, H, W)

        # Apply mask thresholding (only to the 2nd channel)
        image[1] = self.threshold_mask(image[1])

        # Apply random cutout
        # image = self.apply_random_cutout(image)
        gray = image[0].unsqueeze(0)
        mask = image[1].unsqueeze(0)
        stacked_img = torch.cat([gray, mask, gray], dim=0)
        # Apply optional transforms (if provided)
        if self.transform:
            stacked_img = self.transform(stacked_img)
            image[0] = stacked_img[0]
            image[1] = stacked_img[1]

        return image, label.unsqueeze(0)  # Label remains a tensor
