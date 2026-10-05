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
    # TODO: R = quaternion_to_rotation(quat); return R S S^T R^T  -> (N, 3, 3)
    return ...
