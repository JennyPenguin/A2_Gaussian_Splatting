import torch
import matplotlib.pyplot as plt

from fit import fit_2D, evaluate, get_device
from gaussian import covariance_2d
from rasterize import render
from image import load_normalized_image, save_normalized_image

# ---------------------------------------------------------------------
# P5: Fit the 2D Gaussians
# ---------------------------------------------------------------------
densification_runs = 2
gaussian_budget = [1024]
init_N = 128
images = [
    "train_images/coffee",
    # "train_images/astronaut", 
    # "train_images/cat"
]
image_colors = {
    "train_images/coffee": "red",
    "train_images/astronaut": "blue",
    "train_images/cat" : "green"
}

loaded_images = {}
lines_xy = {}
for image in images:
    target = load_normalized_image(image + ".png")
    target = target.to(get_device())
    loaded_images[image] = target
    lines_xy[image] = ([],[])

def train_2D():
    # set seed so can compare consistently
    torch.manual_seed(0)
    for budget in gaussian_budget:
        for image in images:
            # fit_2D will move it to GPU if needed
            target = loaded_images[image]
            for densify in range(densification_runs):
                print(f"Training for image {image} and budget: {budget} and densifcation: {densify}")
                # Start with 1/4 of target so can have good approximation in first
                # iterations before densify and can fully densify at least twice
                start_N = init_N if densify else budget
                mu, log_s, theta, color, op_raw = fit_2D(start_N, target, budget, densification=bool(densify))
                img, psnr = evaluate(mu, log_s, theta, color, op_raw, target)

                print(f"\033[31mpsnr: {psnr} for image {image} and final Gaussian count {mu.shape[0]}\033[0m")

                x, y = lines_xy[image]
                x.append(budget)
                y.append(psnr.item())
                lines_xy[image] = (x, y)

                save_normalized_image(image + str(budget) + "_" + str(densify) + ".png", img)

train_2D()
plt.title("PSNR based on Number of Gaussians")
plt.xlabel("N (Number of Gaussians)")
plt.ylabel("PSNR (db)")
plt.tight_layout()

for image in lines_xy.keys():
    x, y = lines_xy[image]
    plt.plot(x, y, label=f"{image[13:]}", color=image_colors[image])
plt.legend()
plt.savefig('PSNR_P5', dpi=100)

