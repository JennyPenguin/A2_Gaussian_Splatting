import torch

from gaussian import covariance_2d
from rasterize import render
from densification import densify

# run a pass every 200 optimization steps
densify_every = 200
# a target count N from P5 (e.g. 256, 1024, 4096)
max_count = 256    

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = get_device()

def fit_2D(N, target):
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
    log_s = torch.log(0.02 * max(H, W) * torch.ones(N, 2), requires_grad=True, device=device)  
    # (N,) rotation, none at start
    theta = torch.zeros(N,requires_grad=True, device=device)                                   

    # We will apply sigmoid to both color and opacity bc want [0, 1] range
    # (N, 3) sigmoid -> 0.5 gray, is element-wise
    color  = torch.zeros(N, 3, requires_grad=True, device=device)
    # (N,) sigmoid -> ~0.12 opacity
    op_raw = torch.full((N,), -2.0, requires_grad=True, device=device)

    # First list is list of all parameters to update, lr is learning rate
    opt = torch.optim.Adam([mu, log_s, theta, color, op_raw], lr=1e-2)
    grad_mag = torch.zeros((N,), device=device)
    for step in range(1, 2001):
        # By learning log scale and then exp, we guarantee scale is positive
        # Because exp multiply, also means when we add to log scale we multiply
        # original scale so same percentage increase regardless of org size.
        Sigma = covariance_2d(log_s.exp(), theta)
        img   = render(mu, Sigma, color.sigmoid(), op_raw.sigmoid(), depth_order(mu), H, W)
        loss  = ((img - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); 

        # Only look at loss on mean bc positional change implies the gaussian
        # is constantly shifted because there is not enough Gaussians or the
        # Gaussian is too big. Adding other grad leads to too many grad at diff
        # scales.
        grad_mag += torch.norm(mu.grad, dim=-1) / densify_every

        # Adam step does not clear gradients but still put before just in case.
        opt.step()

        # psnr = -10 * torch.log10(loss)

        # Don't densify on last iteration bc won't have time to adapt cloned/
        # split Gaussians afterwards
        if step % densify_every == 0 and step < 2000:
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
