
import numpy as np
from PIL import Image

def overlay_segmentation(original_image_path, masks):
    original_image = Image.open(original_image_path).convert('RGB')
    masks = np.array(masks)
    overlay = np.zeros_like(original_image, dtype=np.uint8)
    for i in range(masks.shape[0]):
        overlay[masks[i] > 0] = [255, 0, 0]  # Red overlay for mask
    combined = Image.blend(original_image, Image.fromarray(overlay), alpha=0.5)
    return combined
