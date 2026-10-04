import torch

from fit import fit_2D, evaluate, get_device
from gaussian import covariance_2d
from rasterize import render
from image import load_normalized_image, save_normalized_image

# ---------------------------------------------------------------------
# P5: Fit the 2D Gaussians
# ---------------------------------------------------------------------
gaussian_budget = [256, 1024, 4096]
images = ["train_images/astronaut.png", 
          "train_images/cat.png", 
          "train_images/coffee.png"]

def train_2D():
    for image in images:
        # fit_2D will move it to GPU if needed
        target = load_normalized_image(image)
        target.to(get_device())
        for budget in gaussian_budget:
            # Start with 1/4 of target so can have good approximation in first
            # iterations before densify and can fully densify at least twice
            init_N = budget // 4
            mu, log_s, theta, color, op_raw = fit_2D(init_N, target, budget)
            img, psnr = evaluate(mu, log_s, theta, color, op_raw, target)
            print(f"psnr: {psnr} for image {image} and final Gaussian count {mu.shape[0]}")
            save_normalized_image(img, image + str(budget))

train_2D()