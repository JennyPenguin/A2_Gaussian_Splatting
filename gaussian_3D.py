import torch
import torch.nn.functional as F

def quaternion_to_rotation(q):
    # do not handle 0 quaternion

    # q: (N, 4) as (w, x, y, z)
    # normalize q, then build R(q) above -> (N, 3, 3)
    q = F.normalize(q, p=2.0, dim=-1)

    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    x2 = x ** 2
    y2 = y ** 2
    z2 = z ** 2
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    e00 = 1.0 - 2.0 * (y2 + z2)
    e01 = 2.0 * (xy - wz) 
    e02 = 2.0 * (xz + wy)
    e10 = 2.0 * (xy + wz)
    e11 = 1.0 - 2.0 * (x2 + z2)
    e12 = 2.0 * (yz - wx)
    e20 = 2.0 * (xz - wy)
    e21 = 2.0 * (yz + wx)
    e22 = 1.0 - 2.0 * (x2 + y2)
    # WIll need to transpose to get (N, 3, 3) instead of (3, 3, N) after so each
    # row is columb
    R = torch.stack([
        torch.stack([e00, e10, e20]),
        torch.stack([e01, e11, e21]),
        torch.stack([e02, e12, e22])
    ])
    return R.transpose(0, -1)

def covariance_3d(scale, quat):
    # scale: (N, 3) positive,  quat: (N, 4)
    # R = quaternion_to_rotation(quat); return R S S^T R^T  -> (N, 3, 3)
    R = quaternion_to_rotation(quat)
    S = torch.diag_embed(scale)
    return R @ S @ S.mT @ R.mT

# R_wc - world to camera rotation
# t - translation
# x_c = R_wcX_world + t = (x_c, y_c, z_c)
# K is camera intrinsics matrix, pinhole camera
# K = [[f_x , 0, c_x]
#   =  [0, f_y, c_y],
#   =  [0, 0, 1]]
# R_wc: (3, 3), K: (3, 3), t:(3,)
def project_gaussian(mu3, Sigma3, R_wc, t, K):
    # mu3: (N, 3) world means,  Sigma3: (N, 3, 3) world covariances
    # tranpose of equation bc each X is a row and not vector here
    mu_cam = mu3 @ R_wc.T + t # world -> camera
    # mu2  =  perspective-project mu_cam with K    (N, 2)
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    # z_c
    depth = mu_cam[:,2]
    # (2, N)
    mu2 = torch.stack([
        mu_cam[:,0] * fx / depth + cx,
        mu_cam[:,1] * fy / depth + cy,
    ], dim=-1)

    N = depth.shape[0]
    d2 = depth ** 2

    # J     = Jacobian of the projection at mu_cam  (N, 2, 3)
    # We use stack so N is in inner dimension. Transpose later so row = col here
    J = torch.stack([
        torch.stack([fx / depth, torch.zeros(N, device=mu3.device)]),
        torch.stack([torch.zeros(N, device=mu3.device), fy / depth]),
        torch.stack([- fx * mu_cam[:,0] / d2, -fy * mu_cam[:,1] / d2])
    ])
    # Transpose to go from (3, 2, N) => (N, 2, 3)
    J = J.transpose(0, -1)
    # Scam  = R_wc @ Sigma3 @ R_wc.T    (N, 3, 3). Scam stands for Scale camera
    Scam = R_wc @ Sigma3 @ R_wc.T
    # Sig2 = J @ Scam @ J.transpose(-1, -2)                (N, 2, 2)
    # J: (N, 3, 3), Scam: (N, 3, 3), J: (N, 3, 3) (transpose each inner matrix)
    Sig2 = J @ Scam @ J.transpose(-1, -2)
    return mu2, Sig2, depth
