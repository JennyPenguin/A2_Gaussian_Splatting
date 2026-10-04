import torch

from gaussian import covariance_2d
from rasterize import render

# run a pass every 200 optimization steps
densify_every = 200
# a target count N from P5 (e.g. 256, 1024, 4096)
max_count = 256     


def fit_2D(N, target):
    # N is # of Gaussians to fit, target is target image, depth_order is 
    W, H, d = target.shape

    assert d == 3 ["Only Support RGB Comparisons"]

    # parameters (leaf tensors, requires_grad=True); a spread-out init, e.g.:
    # Leaf bc we created it directly instead of some differentiable calc

    # (N, 2) spread across the image so they are not all together
    mu = torch.rand(N, 2) * torch.tensor([W, H])    
    mu.requires_grad_(True)      
    # Start with small percentage of image max width/height
    # (N, 2) small blobs, log space so true scale always. positive
    log_s = torch.log(0.02 * max(H, W) * torch.ones(N, 2), requires_grad=True)  
    # (N,) rotation, none at start
    theta = torch.zeros(N,requires_grad=True)                                   

    # We will apply sigmoid to both color and opacity bc want [0, 1] range
    # (N, 3) sigmoid -> 0.5 gray, is element-wise
    color  = torch.zeros(N, 3, requires_grad=True)
    # (N,) sigmoid -> ~0.12 opacity
    op_raw = torch.full((N,), -2.0, requires_grad=True)

    # First list is list of all parameters to update, lr is learning rate
    opt = torch.optim.Adam([mu, log_s, theta, color, op_raw], lr=1e-2)
    for step in range(2000):
        # By learning log scale and then exp, we guarantee scale is positive
        # Because exp multiply, also means when we add to log scale we multiply
        # original scale so same percentage increase regardless of org size.
        Sigma = covariance_2d(log_s.exp(), theta)
        img   = render(mu, Sigma, color.sigmoid(), op_raw.sigmoid(), depth_order(mu), H, W)
        loss  = ((img - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        # psnr = -10 * torch.log10(loss)
    
def depth_order(mu):
    # returns (N,) vector of the ordering of gaussians. For 2D, no depth so just
    # take arange
    N = mu.shape[0]
    return torch.arange(N)