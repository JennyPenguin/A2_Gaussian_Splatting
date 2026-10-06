import torch
from gaussian_3d import covariance_3d, project_gaussian
from rasterize import render
from fit import get_device, make_trainable
from densification_3d import densify_3d

dev = get_device()
DENSIFY = False
# run a pass every 100 optimization steps
densify_every = 100

# Initialize the Gaussians as a coarse random cloud spread through the scene 
# volume. For this dataset the scene sits in world coordinates roughly within a 
# cube with the cameras about 4 units out, so the init above works; a few 
# thousand Gaussians for around 1500 iterations fits it in the minutes range. 
# Keep the resolution (160×160) and count modest so the fit stays fast.

# train_cameras: map from index to R, t, K, H, W, and img (normalized on device)
def fit_3D(train_cameras, iters = 1500):
    N = 4000
    if DENSIFY:
        max_budget = N
        N /= 4
    
    # parameters (leaf tensors, requires_grad=True); example init for this scene:
     # (N, 3)  cloud in ~[-1.5, 1.5]^3
    mu3 = (torch.rand(N, 3, device=dev) * 2.0 - 1.0) *  1.5
    mu3.requires_grad = True
    # (N, 3)  small 3D blobs
    log_s  = torch.log(0.08 * torch.ones(N, 3, device=dev))
    log_s.requires_grad = True
    # (N, 4)  identity rotation (w, x, y, z), w is initialized to 1 and 0 else
    quat   = torch.zeros(N, 4, device=dev); quat[:, 0] = 1.0
    quat.requires_grad = True
    # (N, 3)  sigmoid -> gray
    color  = torch.zeros(N, 3, requires_grad=True, device=dev) 
    # (N,) sigmoid -> low opacity
    op_raw = torch.full((N,), -2.0, requires_grad=True, device=dev)
    grad_mag = torch.zeros((N,), device=dev)
    opt    = torch.optim.Adam([mu3, log_s, quat, color, op_raw], lr=1e-2)

    (train_R, train_t, train_img, H, W, K) = train_cameras
    num_cams = train_R.shape[0]

    # e.g. N = 4000 Gaussians, iters = 1500
    for step in range(1, iters+1):        
        cam_index = torch.randint(0, num_cams, (1,)).item()                
        cam_R  = train_R[cam_index]
        cam_t = train_t[cam_index]
        cam_image = train_img[cam_index]
        # 3D scale + rotation
        Sig3  = covariance_3d(log_s.exp(), quat)
        mu2, Sig2, depth = project_gaussian(mu3, Sig3, cam_R, cam_t, K)
        # front-to-back: nearest (smallest z_c) first
        order = torch.argsort(depth, descending=False)      
        img  = render(mu2, Sig2, color.sigmoid(), op_raw.sigmoid(), order, H, W)
        loss  = ((img - cam_image) ** 2).mean()
        opt.zero_grad(); loss.backward(); 

        if DENSIFY:
            grad_mag += torch.norm(mu3.grad, dim=-1) / densify_every
        
        opt.step()

        if step % 20 == 0:
            psnr = -10 * torch.log10(loss)
            print(f"Step: {step} PSNR: {psnr}")

        if DENSIFY and step % densify_every == 0 and step < iters:
            gaussians = densify_3d((mu3, log_s, quat, color, op_raw), grad_mag, max_budget)

            # Add gradients on new Gaussians
            mu3, log_s, quat, color, op_raw = make_trainable(gaussians)

            # Clear accumulated gradient magnitudes. Need to resize because 
            # Gaussian count likely changed.
            grad_mag = torch.zeros(
                mu3.shape[0],
                device=dev
            )

            # Restart Adam
            opt = torch.optim.Adam([mu3, log_s, quat, color, op_raw], lr=1e-2)

    return mu3, log_s, quat, color, op_raw

# Evaluation, don't calculate any gradients
@torch.no_grad()
def evaluate_3D(mu3, log_s, quat, color, op_raw, cam):
    cam_R, cam_t, cam_image, H, W, K = cam
    Sig3  = covariance_3d(log_s.exp(), quat)
    mu2, Sig2, depth = project_gaussian(mu3, Sig3, cam_R, cam_t, K)
    # front-to-back: nearest (smallest z_c) first
    order = torch.argsort(depth, descending=False)      
    img = render(mu2, Sig2, color.sigmoid(), op_raw.sigmoid(), order, H, W)
    loss  = ((img - cam_image) ** 2).mean()
    psnr = -10 * torch.log10(loss)
    return (img, psnr)
