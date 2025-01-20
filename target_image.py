import numpy as np
import cv2
import torch
from config import image, height, width

target_image = []

def load_image():
    original = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    alpha = original[:, :, 3]
    img = cv2.cvtColor(original[:, :, :3], cv2.COLOR_BGR2RGB) # convert BGR image to RGB
    img = np.dstack((img, alpha))

    # resize the image
    rgba_img = cv2.resize(img, dsize=(width, height))

    return torch.from_numpy(rgba_img / 255.0).float()

def get_target_image():
    global target_image
    if target_image == []:
        target_image = load_image()
    return target_image