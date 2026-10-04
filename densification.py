import torch

# densify Gaussian i if g_i > grad_threshold
grad_threshold = 2e-7
# clone if max scale <= 2% of image width, else split
 # each split child gets (parent scale / split_scale)
size_threshold = 0.02
split_scale = 1.6         
# remove Gaussian i if its opacity < this
prune_opacity  = 0.005  

# run inside the P3 training loop, every densify_every steps
# Don't compute gradients for densificiation
@torch.no_grad()
def densify(gaussians, grad_mag, budget, W, H):
    # gaussians: all tensors needed to define Gaussians. Each of shape (N,_)
    # grad_mag: per-Gaussian g_i accumulated since the last pass
    # budget: total amount of Gaussians allowed
    # W, H: width and height of image (to compare scaling)
    mu, log_s, theta, color, op_raw = gaussians

    # use same device as other variables
    device = mu.device

    N = mu.shape[0]
    indices = torch.arange(N, device=device)

    o = op_raw.sigmoid()

    # prune Gaussians with opacity < prune_opacity
    # the reason we prune first is because we have a maximum budget so to free
    # up budget for splitting/cloning, we prune first.
    prune_mask = o >= prune_opacity
    # Only keep indices that survive pruning
    pruned_indices = indices[prune_mask]
    pruned_N = pruned_indices.shape[0]
    print(f"Number of prune: {N - pruned_N}/{N}")
    mu_p, log_s_p, theta_p, color_p, op_raw_p = mu[pruned_indices], log_s[pruned_indices], theta[pruned_indices], color[pruned_indices], op_raw[pruned_indices]
    grad_mag_p = grad_mag[pruned_indices]

    # Indices for for pruned version
    indices_p = torch.arange(pruned_N, device=device)

    # After pruning, we calculate true remaining budget left for split/clone
    remain_B = budget - pruned_N
    assert remain_B >= 0, "Densification allowed above budget!"
    
    # dense = grad_mag (without pruned) > grad_threshold (under-fit Gaussians)
    mask_dense = grad_mag_p > grad_threshold
    dense_g = grad_mag_p[mask_dense]
    print(
        f"grad_mag: "
        f"min={grad_mag.min().item():.5e}, "
        f"mean={grad_mag.mean().item():.5e}, "
        f"max={grad_mag.max().item():.5e}"
    )
    print(f"Number of dense: {dense_g.shape[0]}/{N}")
    dense_i = indices_p[mask_dense]

    # keep the total count <= budget, don't care what top Gaussian losses are
    # if has less gaussians than budget, passing in remain_B directly will fail
    K = min(remain_B, dense_g.shape[0])

    # No Gaussians to change or no budget left
    if K == 0:
        return (mu_p, log_s_p, theta_p, color_p, op_raw_p)

    _, top_i = torch.topk(dense_g, k=K)
    dense_top_i = dense_i[top_i]

    mask_nd = torch.ones(pruned_N, dtype=torch.bool, device=device)
    mask_nd[dense_top_i] = False
    mu_nd, log_s_nd, theta_nd, color_nd, op_raw_nd = mu_p[mask_nd], log_s_p[mask_nd], theta_p[mask_nd], color_p[mask_nd], op_raw_p[mask_nd]
    # These are the final Gaussians to split/clone
    mask_d = ~mask_nd
    mu_d, log_s_d, theta_d, color_d, op_raw_d = mu_p[mask_d], log_s_p[mask_d], theta_p[mask_d], color_p[mask_d], op_raw_p[mask_d]
    s_d = log_s_d.exp()

    size_thres = size_threshold * max(W, H)
    
    # clone = dense & (max_scale <= size_threshold)  (duplicate in place)
    # don't need to do  because values did not change, can just concat

    # split = dense & (max_scale >  size_threshold)  
    # (2 children, scale / split_scale)
    mask_split = torch.max(s_d, dim=-1).values > size_thres
    s_d[mask_split] /= split_scale
    # log is base e default
    log_s_d = torch.log(s_d)

    mu_f, log_s_f, theta_f, color_f, op_raw_f = torch.concat([mu_nd, mu_d, mu_d]), torch.concat([log_s_nd, log_s_d, log_s_d]), torch.concat([theta_nd, theta_d, theta_d]), torch.concat([color_nd, color_d, color_d]), torch.concat([op_raw_nd, op_raw_d, op_raw_d])

    final_N = mu_f.shape[0]
    assert final_N <= budget, "Densification allowed above budget!"
 
    return (mu_f, log_s_f, theta_f, color_f, op_raw_f)