import torch

from gaussian import gaussian_weight

# Faster version by chatGPT with cumprod
# Naive version writte by me below
def render(mu, Sigma, color, opacity, order, H, W):
    # P = H * W

    xy = pixel_grid(H, W).to(mu.device)       # (P, 2)

    # Gaussian falloff at every pixel
    w = gaussian_weight(xy, mu, Sigma)         # (P, N)

    # Per-pixel Gaussian alpha
    alpha = opacity[None, :] * w               # (P, N)

    # Sort front -> back
    alpha = alpha[:, order]                    # (P, N)
    color = color[order]                       # (N, 3)

    # T_i = product of (1-alpha_j) for j < i
    T = torch.cumprod(
        torch.cat([
            torch.ones(
                alpha.shape[0],
                1,
                device=alpha.device,
                dtype=alpha.dtype
            ),
            1.0 - alpha
        ], dim=1),
        dim=1
    )[:, :-1]                                  # (P, N)

    # Premultiplied contribution from every Gaussian
    weights = T * alpha                        # (P, N)

    # Sum all Gaussian colors
    C = weights @ color                        # (P, 3)

    return C.reshape(H, W, 3)

def render_naive(mu, Sigma, color, opacity, order, H, W):
    # color: (N, 3),  opacity: (N,) in [0, 1],  order: indices sorted front -> back

    # Colors are straight (non-premultiplied) RGB in [0, 1]
    # order lists the Gaussians front to back. In 2D there is no real depth, so any fixed order works (plain index order, torch.arange(N), is fine) and the optimizer adapts the colors and opacities to it. In 3D you will instead sort by camera-space depth (P7)

    xy = pixel_grid(H, W)                     # (H * W, 2)
                                              # H * W = P
    xy = xy.to(mu.device)
    w  = gaussian_weight(xy, mu, Sigma)       # (P, N)  from P1 (all gaussian 
                                              # contribution at every pixel)
    alpha = opacity[None, :] * w              # (P, N), w is falloff
    # Start with black (no) color
    C = torch.zeros(H * W, 3, device=mu.device)
    # Start with everything passing through
    T = torch.ones(H * W, device=mu.device)
    
    # A dense per-pixel evaluation over all Gaussians is fine at this scale. If it is slow, cap each Gaussian’s influence to a local window around its center rather than the whole image.
    for i in order:                           # front to back
        a = alpha[:, i] # (P,)
        c = color[i, :] # (,3)
        # a * c is premultiplied RGB dim (P, 3)
        premult = a[:,None] @ c[None,:]

        # C and T compositing here. C dim (P, 3), T dim (P,)
        # DO NOT do += or *= bc those are in-place and corrupt backprob
        # C = C + is out of place. Create new tensor and repoint C to it
        C = C + T[:,None] * premult 
        T = T * (1.0 - a) # Whatever passed through
    return C.reshape(H, W, 3) # result image is alpha-premultiply over black bg

def pixel_grid(H, W):
    # Generates pixel coordinates (to help calculate falloff from Gaussian)
    y_coords = torch.arange(H, dtype=torch.float32)
    x_coords = torch.arange(W, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
    # Shape (W, H, 2)
    coords = torch.stack([grid_x, grid_y], dim=-1)
    # Reshape to single pixels vector
    coords = coords.reshape(W * H, 2)
    # Adds 0.5 for pixel center
    coords += 0.5
    return coords