import torch

from fit import fit_2D, evaluate, get_device
from gaussian import covariance_2d
from rasterize import render
from image import load_normalized_image, save_normalized_image

# ---------------------------------------------------------------------
# P5: Fit the 2D Gaussians
# ---------------------------------------------------------------------
gaussian_budget = [4096]
init_N = 4096
images = ["train_images/coffee",
          "train_images/astronaut", 
          "train_images/cat"]

def train_2D():
    # set seed so can compare consistently
    torch.manual_seed(0)
    for image in images:
        # fit_2D will move it to GPU if needed
        target = load_normalized_image(image + ".png")
        target = target.to(get_device())
        for budget in gaussian_budget:
            print(f"Training for image {image} and budget: {budget}")
            # Start with 1/4 of target so can have good approximation in first
            # iterations before densify and can fully densify at least twice
            mu, log_s, theta, color, op_raw = fit_2D(init_N, target, budget)
            img, psnr = evaluate(mu, log_s, theta, color, op_raw, target)
            print(f"\033[31mpsnr: {psnr} for image {image} and final Gaussian count {mu.shape[0]}\033[0m")
            save_normalized_image(image + str(budget) + ".png", img)

train_2D()