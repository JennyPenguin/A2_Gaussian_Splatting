import torch
import time

from gaussian import covariance_2d
from rasterize import render
from densification import densify
from image import save_normalized_image

# run a pass every 200 optimization steps
densify_every = 200

train_iters = 2000

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = get_device()

def fit_2D(N, target, max_count, densification=True):
    assert N <= max_count, "Started with more Gaussians than allowed!"

    target = target.to(device)
    # N is # of Gaussians to fit, target is target image, depth_order is 
    H, W, d = target.shape

    assert d == 3, "Only Support RGB Comparisons"

    # parameters (leaf tensors, requires_grad=True); a spread-out init, e.g.:
    # Leaf bc we created it directly instead of some differentiable calc

    # (N, 2) spread across the image so they are not all together
    mu = torch.rand(N, 2, device=device) * torch.tensor([W, H], device=device)    
    mu.requires_grad_(True)  
    # Start with small percentage of image max width/height
    # (N, 2) small blobs, log space so true scale always. positive
    log_s = torch.log(0.02 * max(H, W) * torch.ones(N, 2, device=device))  
    log_s.requires_grad_(True)
    # (N,) rotation, none at start
    theta = torch.zeros(N, requires_grad=True, device=device)                                   

    # We will apply sigmoid to both color and opacity bc want [0, 1] range
    # (N, 3) sigmoid -> 0.5 gray, is element-wise
    color  = torch.zeros(N, 3, requires_grad=True, device=device)
    # (N,) sigmoid -> ~0.12 opacity
    op_raw = torch.full((N,), -2.0, requires_grad=True, device=device)

    # First list is list of all parameters to update, lr is learning rate
    opt = torch.optim.Adam([mu, log_s, theta, color, op_raw], lr=1e-2)
    grad_mag = torch.zeros((N,), device=device)
    for step in range(1, train_iters):
        # torch.mps.synchronize()
        # t0 = time.perf_counter()

        # By learning log scale and then exp, we guarantee scale is positive
        # Because exp multiply, also means when we add to log scale we multiply
        # original scale so same percentage increase regardless of org size.
        # if step > 1800:
        #     check_finite("mu", mu, step)
        #     check_finite("log_s", log_s, step)
        #     check_finite("theta", theta, step)
        #     check_finite("color", color, step)
        #     check_finite("op_raw", op_raw, step)
        s = log_s.exp()
        # if step > 1800:
        #     check_finite("scale", s, step)
        Sigma = covariance_2d(s, theta)
        # if step > 1800:
        #     check_finite("Sigma", Sigma, step)
        #     print(
        #         f"scale: min={s.min().item():.3e}, "
        #         f"max={s.max().item():.3e}"
        #     )

        #     print(
        #         f"log_s: min={log_s.min().item():.3e}, "
        #         f"max={log_s.max().item():.3e}"
        #     )
    
        img   = render(mu, Sigma, color.sigmoid(), op_raw.sigmoid(), depth_order(mu), H, W)

        # if step > 1800:
        #     print("After render")
        #     check_finite("img", img, step)
        #     check_finite("loss", loss, step)
        #     check_finite("mu.grad", mu.grad, step)
        #     check_finite("log_s.grad", log_s.grad, step)
        #     check_finite("theta.grad", theta.grad, step)
        #     check_finite("color.grad", color.grad, step)
        #     check_finite("op_raw.grad", op_raw.grad, step)

        # torch.mps.synchronize()
        # t1 = time.perf_counter()

        loss  = ((img - target) ** 2).mean()
        
        opt.zero_grad(); loss.backward(); 

        # if step > 1800:
        #     print("After backwards")
        #     check_finite("loss", loss, step)
        #     check_finite("mu.grad", mu.grad, step)
        #     check_finite("log_s.grad", log_s.grad, step)
        #     check_finite("theta.grad", theta.grad, step)
        #     check_finite("color.grad", color.grad, step)
        #     check_finite("op_raw.grad", op_raw.grad, step)

        # torch.mps.synchronize()
        # t2 = time.perf_counter()

        # Only look at loss on mean bc positional change implies the gaussian
        # is constantly shifted because there is not enough Gaussians or the
        # Gaussian is too big. Adding other grad leads to too many grad at diff
        # scales.
        if densification:
            grad_mag += torch.norm(mu.grad, dim=-1) / densify_every

        opt.step()

        # if step > 1800:
        #     print("After loss")
        #     check_finite("loss", loss, step)
        #     check_finite("mu.grad", mu.grad, step)
        #     check_finite("log_s.grad", log_s.grad, step)
        #     check_finite("theta.grad", theta.grad, step)
        #     check_finite("color.grad", color.grad, step)
        #     check_finite("op_raw.grad", op_raw.grad, step)

        # torch.mps.synchronize()
        # t3 = time.perf_counter()

        # print("forward:", t1 - t0)
        # print("backward:", t2 - t1)
        # print("optimizer:", t3 - t2)

        # Don't densify on last iteration bc won't have time to adapt cloned/
        # split Gaussians afterwards
        if step % densify_every == 0 and step < 2000:
            # save_normalized_image(f"debug/{step}.png", img.detach())
            psnr = -10 * torch.log10(loss)
            print(f"Step {step} PSNR: {psnr}")

            if densification:
                gaussians = densify((mu, log_s, theta, color, op_raw), grad_mag,max_count, W, H)

                # Add gradients on new Gaussians
                mu, log_s, theta, color, op_raw = make_trainable(gaussians)

                # Clear accumulated gradient magnitudes. Need to resize because 
                # Gaussian count likely changed.
                grad_mag = torch.zeros(
                    mu.shape[0],
                    device=device
                )

                # Restart Adam
                opt = torch.optim.Adam([mu, log_s, theta, color, op_raw], lr=1e-2)
    return mu, log_s, theta, color, op_raw
    
def depth_order(mu):
    # returns (N,) vector of the ordering of gaussians. For 2D, no depth so just
    # take arange
    N = mu.shape[0]
    return torch.arange(N, device=mu.device)

def make_trainable(gaussians):
    return tuple(
        x.detach().requires_grad_(True)
        for x in gaussians
    )

# Evaluation, don't calculate any gradients
@torch.no_grad()
def evaluate(mu, log_s, theta, color, op_raw, target):
    H, W, _ = target.shape
    Sigma = covariance_2d(log_s.exp(), theta)
    img   = render(mu, Sigma, color.sigmoid(), op_raw.sigmoid(), depth_order(mu), H, W)
    loss  = ((img - target) ** 2).mean()
    psnr = -10 * torch.log10(loss)
    return (img, psnr)


def check_finite(name, x, step):
    if x is None:
        print(f"{name} is None at step {step}")
        return True

    if not torch.isfinite(x).all():
        print(f"NON-FINITE {name} at step {step}")
        assert False

    return True
