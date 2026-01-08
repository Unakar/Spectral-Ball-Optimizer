from typing import Any, Literal

import torch
from absl import logging



_COEFFICIENT_SETS = {
    "simple": [
        (3.4445, -4.7750, 2.0315),
    ],
    "quintic": [
        # optimized for a quintic iteration.
        # Source: https://leloykun.github.io/ponder/muon-opt-coeffs/#how-do-we-optimize-the-coefficients
        # Numbers from: https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt_medium.py#L44
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ],
    "polar_express": [
        # Polar Express iteration from: https://arxiv.org/abs/2505.16932
        (8.2051, -22.9019, 16.4607),
        (4.0664, -2.8612, 0.5184),
        (3.9096, -2.8234, 0.5250),
        (3.2856, -2.4153, 0.4853),
        (2.2779, -1.6198, 0.3985),
        (1.8726, -1.2307, 0.3585),
        (1.8564, -1.2132, 0.3568),
        (1.8750, -1.2500, 0.3750),
    ],
    "aol": [
        # from https://github.com/thib-s/flash-newton-schulz/blob/main/newton_schulz_triton.py#L511
        (4.0098, -7.0585, 2.4635),
        (3.4585, -5.5479, 2.5959),
        (2.7573, -3.2939, 1.4254),
        (2.7215, -3.0494, 1.3169),
    ],
}


def distributed_normalize_p2(x: torch.Tensor, eps: float, group: torch.distributed.ProcessGroup) -> torch.Tensor:
    """Normalize a tensor in a distributed way."""
    x_sq_sum = (x * x).sum()
    torch.distributed.all_reduce(x_sq_sum, op=torch.distributed.ReduceOp.SUM, group=group)
    return x / torch.sqrt(x_sq_sum).clamp_min(eps)


def newton_schulz_step(
    X: torch.Tensor, a: float, b: float, c: float, tp_group: torch.distributed.ProcessGroup | None = None
) -> torch.Tensor:
    A = X @ X.mT
    if tp_group is not None:
        torch.distributed.all_reduce(A, op=torch.distributed.ReduceOp.SUM, group=tp_group)
    B = torch.addmm(A, A, A, alpha=c, beta=b)
    X = torch.addmm(X, B, X, alpha=1.0, beta=a)
    return X


def newton_schulz(
    x: torch.Tensor,
    steps: int,
    coefficient_type: str = "quintic",
    custom_coefficient_sets: list[tuple[float, float, float]] | None = None,
    eps: float = 1e-7,
    transpose: bool | None = None,
    tp_group: torch.distributed.ProcessGroup | None = None,
    use_syrk: bool = False,
) -> torch.Tensor:
    # Muon is not for 1d parameters
    if x.ndim < 2:
        raise ValueError("Input tensor x must have at least 2 dimensions since Muon is not for 1d parameters.")
    if x.dtype != torch.float32:
        raise ValueError(f"Input tensor x must be in float32, got {x.dtype}")

    # transpose tensor to perform whitening on the smaller dimension
    if transpose is None:
        transpose = x.size(-2) > x.size(-1)
    if transpose:
        x = x.mT

    # Ensure spectral norm is at most 1
    if tp_group is not None:
        X = distributed_normalize_p2(x, eps, tp_group)
    else:
        X = torch.nn.functional.normalize(x, p=2, dim=(-2, -1), eps=eps)

    if coefficient_type in _COEFFICIENT_SETS:
        coefficient_sets = _COEFFICIENT_SETS[coefficient_type]
    elif coefficient_type == "custom":
        if custom_coefficient_sets is None:
            raise ValueError("custom_coefficient_sets must be provided when coefficient_type is 'custom'.")
        coefficient_sets = custom_coefficient_sets
    else:
        raise ValueError(f"Invalid coefficient type: {coefficient_type}")

    if steps % len(coefficient_sets) != 0:
        raise ValueError(f"steps ({steps}) must be multiple of len(coefficient_sets) ({len(coefficient_sets)}).")

    ns_step_fn = newton_schulz_step
    # Perform the NS iterations
    if torch.get_float32_matmul_precision() == "medium":
        # PyTorch doesn't really have FP32 I/O BF16 compute kernels for precision "medium"
        # We explicitly convert to BF16 and back to FP32.
        # NOTE: There is a small difference to calling FP32 I/O BF16 compute kernels because the final result
        # is converted to BF16 before converting back to FP32. The rest should be the same as long as epilogue
        # is always in FP32.
        X = X.to(torch.bfloat16)
        logging.log_first_n(logging.INFO, "Using BF16 I/O kernels for Newton-Schulz iteration.", 1)

    for i in range(steps):
        a, b, c = coefficient_sets[i % len(coefficient_sets)]
        X = ns_step_fn(X, a, b, c, tp_group=tp_group)

    # Convert back to FP32. This is a noop if X is already in FP32.
    X = X.to(torch.float32)

    # undo transpose if necessary
    if transpose:
        X = X.mT
    return X


def msign_svd(x: torch.Tensor, eps: float = 1e-7, transpose: bool | None = None) -> torch.Tensor:
    """Compute matrix sign function using SVD: sign(X) = U @ V.T where X = U @ S @ V.T"""
    if x.ndim < 2:
        raise ValueError("Input tensor x must have at least 2 dimensions.")
    if x.dtype != torch.float32:
        raise ValueError(f"Input tensor x must be in float32, got {x.dtype}")

    # transpose tensor to perform whitening on the smaller dimension
    if transpose is None:
        transpose = x.size(-2) > x.size(-1)
    if transpose:
        x = x.mT

    # Ensure spectral norm is at most 1
    X = torch.nn.functional.normalize(x, p=2, dim=(-2, -1), eps=eps)

    # Compute SVD
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)

    # Matrix sign is U @ V.T
    result = U @ Vh

    # undo transpose if necessary
    if transpose:
        result = result.mT
    return result


def compute_rms_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute RMS (Root Mean Square) error between two tensors."""
    return torch.sqrt(torch.mean((a - b) ** 2)).item()


def test_msign_methods():
    """Test and compare quintic(5), polar_express(8), and SVD methods for computing msign."""
    print("=" * 80)
    print("Testing Matrix Sign Function Methods")
    print("=" * 80)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Test different matrix sizes
    test_sizes = [
        (512, 512),
        (1024, 512),
        (2048, 1024),
    ]

    for m, n in test_sizes:
        print(f"\nTesting matrix size: {m} x {n}")
        print("-" * 80)

        # Create random test matrix
        x = torch.randn(m, n, dtype=torch.float32)

        # Compute msign using SVD (ground truth)
        msign_svd_result = msign_svd(x.clone())

        # Compute msign using quintic (5 steps)
        msign_quintic = newton_schulz(x.clone(), steps=5, coefficient_type="quintic")

        # Compute msign using polar_express (8 steps)
        msign_polar = newton_schulz(x.clone(), steps=8, coefficient_type="polar_express")

        # Compute RMS errors
        rms_quintic = compute_rms_error(msign_quintic, msign_svd_result)
        rms_polar = compute_rms_error(msign_polar, msign_svd_result)

        print(f"Quintic (5 steps) RMS Error:        {rms_quintic:.6e}")
        print(f"Polar Express (8 steps) RMS Error:  {rms_polar:.6e}")

        # Check orthogonality (X.T @ X should be identity)
        print("\nOrthogonality check (|| X.T @ X - I ||_F):")
        identity = torch.eye(min(m, n), dtype=torch.float32)

        ortho_svd = torch.norm(msign_svd_result.mT @ msign_svd_result - identity, p='fro').item()
        ortho_quintic = torch.norm(msign_quintic.mT @ msign_quintic - identity, p='fro').item()
        ortho_polar = torch.norm(msign_polar.mT @ msign_polar - identity, p='fro').item()

        print(f"SVD:                                {ortho_svd:.6e}")
        print(f"Quintic (5 steps):                  {ortho_quintic:.6e}")
        print(f"Polar Express (8 steps):            {ortho_polar:.6e}")

    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)


if __name__ == "__main__":
    test_msign_methods()