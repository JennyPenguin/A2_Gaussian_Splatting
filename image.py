from PIL import Image
import torch
import numpy as np

def load_normalized_image(path: str):
    img = Image.open(path)
    # Make sure it's RGB and range [0, 1]
    img = img.convert("RGB")
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0)

def save_normalized_image(path: str, img):
    # img is originally pytorch tensor so we convert to numpy for saving
    img = img.numpy()
    img = (np.round(img * 255.0)).astype(np.uint8)
    Image.fromarray(img).save(path)