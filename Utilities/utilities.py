import SimpleITK as sitk
from PIL import Image
import random
import os
import pydicom
import numpy as np
import torch

def load_mask(mask_path):
    mask = sitk.ReadImage(mask_path)
    return sitk.GetArrayFromImage(mask)[0][: 450, 200: 1300]

def get_bounding_box(mask, padding=10):
    """
    Compute the bounding box around the nonzero mask region with optional padding.
    
    Args:
        mask (numpy array): Binary mask where nonzero pixels represent the object.
        padding (int): Extra pixels to add around the bounding box.
    
    Returns:
        (y_min, y_max, x_min, x_max): Cropped bounding box coordinates.
    """

    coords = np.argwhere(mask > 0)  

    if coords.size == 0:
        return None  # No object found

    # Get bounding box coordinates
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)


    y_min = max(y_min - padding, 0)
    x_min = max(x_min - padding, 0)
    y_max = min(y_max + padding, mask.shape[0])
    x_max = min(x_max + padding, mask.shape[1])

    return y_min, y_max, x_min, x_max

def load_dicom(dicom_path):
    dicom_data = pydicom.dcmread(dicom_path)
    return dicom_data.pixel_array[: 450, 200: 1300,0]

def dicom_to_jpg(image):
    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    img = Image.fromarray(image)  # Convert to PIL Image
    return img.convert('L')

def anno_to_yolo(anno_file, height, width):
    mask = load_mask(anno_file)
    ymin, ymax, xmin, xmax = get_bounding_box(mask, padding=10)
    
    center_x = (xmin + xmax) / (2 * width)
    center_y = (ymin + ymax) / (2 * height)
    width = (xmax - xmin) / width
    height = (ymax - ymin) / height

    return [center_x, center_y, width, height]

def load_model(model, optimizer=None, scheduler=None, path='./checkpoint.pth'):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        optimizer = None
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        scheduler = None
    epoch = checkpoint['epoch']
    return model, optimizer, scheduler, epoch

def save_model(model,path):
    torch.save(
        {'model_state_dict': model.state_dict()},path)

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
