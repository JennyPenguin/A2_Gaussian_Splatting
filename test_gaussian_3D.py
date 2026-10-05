# test_gaussian_3d.py

import torch
import pytest

from gaussian_3D import quaternion_to_rotation

def test_output_shape():
    """(N, 4) quaternions should produce (N, 3, 3) matrices."""
    q = torch.randn(10, 4)

    R = quaternion_to_rotation(q)

    assert R.shape == (10, 3, 3)


def test_identity_quaternion():
    """q = (1, 0, 0, 0) should give the identity rotation."""
    q = torch.tensor([
        [1.0, 0.0, 0.0, 0.0]
    ])

    R = quaternion_to_rotation(q)

    expected = torch.eye(3).unsqueeze(0)

    assert torch.allclose(R, expected, atol=1e-6)


def test_unnormalized_identity_quaternion():
    """
    Quaternion magnitude should not matter because the function
    normalizes q internally.
    """
    q = torch.tensor([
        [5.0, 0.0, 0.0, 0.0]
    ])

    R = quaternion_to_rotation(q)

    expected = torch.eye(3).unsqueeze(0)

    assert torch.allclose(R, expected, atol=1e-6)


def test_normalization_invariance():
    """
    q and k*q should represent the same rotation for positive k.
    """
    q = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, -1.0, 0.5, 3.0]
    ])

    R1 = quaternion_to_rotation(q)
    R2 = quaternion_to_rotation(7.0 * q)

    assert torch.allclose(R1, R2, atol=1e-5)


def test_known_180_degree_x_rotation():
    """
    q = (0, 1, 0, 0) represents a 180-degree rotation around x.

    Expected:
        x ->  x
        y -> -y
        z -> -z
    """
    q = torch.tensor([
        [0.0, 1.0, 0.0, 0.0]
    ])

    R = quaternion_to_rotation(q)

    expected = torch.tensor([[
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0]
    ]])

    assert torch.allclose(R, expected, atol=1e-6)


def test_rotation_matrix_is_orthogonal():
    """
    A valid rotation matrix must satisfy:

        R R^T = I
    """
    q = torch.randn(20, 4)

    R = quaternion_to_rotation(q)

    product = R @ R.mT

    expected = (
        torch.eye(3, dtype=R.dtype, device=R.device)
        .unsqueeze(0)
        .expand(20, -1, -1)
    )

    assert torch.allclose(product, expected, atol=1e-5)


def test_rotation_matrix_determinant():
    """A proper 3D rotation matrix should have determinant +1."""
    q = torch.randn(20, 4)

    R = quaternion_to_rotation(q)

    det = torch.linalg.det(R)

    expected = torch.ones_like(det)

    assert torch.allclose(det, expected, atol=1e-5)


def test_batch_independence():
    """
    Batched evaluation should give the same answer as evaluating
    each quaternion separately.
    """
    q = torch.randn(8, 4)

    R_batch = quaternion_to_rotation(q)

    for i in range(q.shape[0]):
        R_single = quaternion_to_rotation(q[i:i+1])

        assert torch.allclose(
            R_batch[i:i+1],
            R_single,
            atol=1e-6
        )


def test_preserves_dtype():
    """Output should retain q's floating-point dtype."""
    q = torch.randn(5, 4, dtype=torch.float64)

    R = quaternion_to_rotation(q)

    assert R.dtype == q.dtype


def test_preserves_device():
    """Output must remain on the same device as q."""
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    q = torch.randn(5, 4, device=device)

    R = quaternion_to_rotation(q)

    assert R.device == q.device


def test_gradient_propagation():
    """
    quaternion_to_rotation must preserve the autograd graph.
    """
    q = torch.randn(
        5, 4,
        requires_grad=True
    )

    R = quaternion_to_rotation(q)

    assert R.requires_grad
    assert R.grad_fn is not None

    # Don't use R.sum() here: because of rotation-matrix symmetry,
    # that particular loss can accidentally produce zero gradients
    # for some quaternions.
    weights = torch.arange(
        1,
        R.numel() + 1,
        dtype=R.dtype,
        device=R.device
    ).reshape_as(R)

    loss = (R * weights).sum()
    loss.backward()

    assert q.grad is not None
    assert q.grad.shape == q.shape
    assert torch.isfinite(q.grad).all()