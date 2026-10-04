import torch

# Scale & Theta should have dtype=torch.float32
def covariance_2d(scale, theta):
    # scale: (N, 2) positive,  theta: (N,) radians
    
    # Build R(theta) and S = diag(scale), return Sigma = R S S^T R^T  -> (N, 2, 2)

    # Ensure scale & theta are both tensors but they really should be already 
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, dtype=torch.float32)

    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, dtype=torch.float32)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    # Standard 2D Rotation Matrix
    # Counterclosewise but don't matter bc gradient will adapt anyway
    # R has shape (2, 2, N), will transpose later so each row is actually col
    R = torch.stack([
        torch.stack([cos_t, sin_t]),
        torch.stack([-sin_t, cos_t])
    ])
    # Transpose bc we want N to be outermost layer. Now shape (N, 2, 2)
    R = R.transpose(0, -1)

    # S has shape (N, 2, 2) -> replace each vector with diagonal matrix
    S = torch.diag_embed(scale)
    # .mT is batch response
    # Square scale because variance
    return R @ S @ S.mT @ R.mT

def gaussian_weight(xy, mu, Sigma):
    # (P, N)
    dx = xy[:, None, 0] - mu[None, :, 0]
    dy = xy[:, None, 1] - mu[None, :, 1]

    Sigma_inv = torch.linalg.inv(Sigma)

    a = Sigma_inv[:, 0, 0]
    b = Sigma_inv[:, 0, 1]
    c = Sigma_inv[:, 1, 1]

    exponent = (
        a[None, :] * dx.square()
        + 2.0 * b[None, :] * dx * dy
        + c[None, :] * dy.square()
    )

    return torch.exp(-0.5 * exponent)

def gaussian_weight_naive(xy, mu, Sigma):
    # xy: (P, 2) pixel coords,  mu: (N, 2),  Sigma: (N, 2, 2)
    # w[p, n] = exp(-0.5 (xy_p - mu_n)^T Sigma_n^-1 (xy_p - mu_n))
    
    # N is # of Gaussians, P is # of pixels
    # Want to evaluate each Gaussian at every pixel
    # Get xy_p - mu_n first, need to transform size to broadcast
    # Putting None inserts a new dimesion of 1 at this position
    # Shape (P, N, 2)
    diff = xy[:, None, :] - mu[None, :, :]
    # Shape (N, 2, 2)
    Sigma_inv = torch.linalg.inv(Sigma)

    # For each (P, N) pair, we want to compute the diff^T * Sigma_inv * diff
    # Convert sigma_inv to (1, N, 2, 2) for broadcasting (actually can skip bc default behavior). To get col version turn diff to (P, N, 2, 1). To get row version turn diff into (P, N, 1, 2). Final result is (P, N, 1, 1)
    exponent = diff[:,:,None,:] @ Sigma_inv[None,:,:,:] @ diff[:,:,:,None]
    # Squeeze to get rid of (1, 1) dimension at the end
    exponent = exponent.squeeze(dim=(-1, -2))
    result = torch.exp(-0.5 * exponent)

    return result 


## Small Self Checks
# # N = 3
# theta = torch.tensor([0, 1, 2], dtype=torch.float32)
# scale = torch.tensor([[1, 2], [5, 6], [3, 3]], dtype=torch.float32)
# C = covariance_2d(scale, theta)
# print(C.shape)
# print(C)

# # P = 4
# xy = torch.tensor([[0, 1], [1, 0], [0, 0], [1, 1]], dtype=torch.float32)
# mu = torch.tensor([[0, 0],[1, 1],[2, 2]], dtype=torch.float32)
# G = gaussian_weight(xy, mu, C)
# print(G.shape)




