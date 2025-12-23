"""
Numerical verification tool for f(lambda) = <Theta, msign(G + lambda * Theta)>.

This tool tests the f function from spectral_ball_utils.py to verify:
1. f is monotonically increasing
2. Find the zero point of f (suspected to be near 0)
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import brentq

# ============================================================================
# Core functions copied from spectral_ball_utils.py
# ============================================================================


def _muon_newton_schulz_step(
    X: torch.Tensor, a: float, b: float, c: float
) -> torch.Tensor:
    """One Newton-Schulz iteration: X ← a·X + X·(b·A + c·A²) where A = X·X^T."""
    A = X @ X.mT
    B = torch.addmm(A, A, A, alpha=c, beta=b)
    X = torch.addmm(X, B, X, alpha=1.0, beta=a)
    return X


def msign(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Matrix sign via Newton-Schulz with Polar-Express coefficients (8 steps)."""
    if G.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions.")
    if G.dtype != torch.float32:
        raise ValueError(f"Input tensor G must be in float32")

    transpose = G.size(-2) > G.size(-1)
    X = G.mT if transpose else G
    X = torch.nn.functional.normalize(X, p=2, dim=(-2, -1), eps=1e-7)
    X = X.to(torch.bfloat16)

    # Polar-Express coefficients
    coeffs = [
        (8.2051, -22.9019, 16.4607),
        (4.0664, -2.8612, 0.5184),
        (3.9096, -2.8234, 0.5250),
        (3.2856, -2.4153, 0.4853),
        (2.2779, -1.6198, 0.3985),
        (1.8726, -1.2307, 0.3585),
        (1.8564, -1.2132, 0.3568),
        (1.8750, -1.2500, 0.3750),
    ]

    for i in range(steps):
        a, b, c = coeffs[i % 8]
        X = _muon_newton_schulz_step(X, a, b, c)

    X = X.to(torch.float32)
    return X.mT if transpose else X


@torch.no_grad()
def inner_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Frobenius inner product <a, b>."""
    return (a * b).sum()


@torch.no_grad()
def compute_phi(
    G: torch.Tensor, Theta: torch.Tensor, lambda_value: float, msign_steps: int = 8
) -> torch.Tensor:
    """φ(λ) = msign(G + λ·Θ)."""
    z = G + lambda_value * Theta
    Phi = msign(z, steps=msign_steps)
    return Phi


@torch.no_grad()
def compute_f(
    G: torch.Tensor, Theta: torch.Tensor, lambda_value: float, msign_steps: int = 8
) -> float:
    """f(λ) = <Θ, msign(G + λ·Θ)>."""
    Phi = compute_phi(G, Theta, lambda_value, msign_steps)
    f_value = float(inner_product(Theta, Phi).item())
    return f_value


# ============================================================================
# Test matrix generation
# ============================================================================


def generate_test_matrices(
    m: int,
    n: int,
    mean: float = 0.0,
    std: float = 0.02,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate test matrices G and Theta.

    Args:
        m: Number of rows
        n: Number of columns
        mean: Mean of the normal distribution
        std: Standard deviation of the normal distribution
        seed: Random seed for reproducibility
        device: Device to generate matrices on (CPU or CUDA)

    Returns:
        G: Gradient tensor (normalized, as in the source code)
        Theta: Direction tensor (rank-1 from SVD of W)
    """
    # Generate G (independent random matrix)
    torch.manual_seed(seed)
    G_raw = torch.randn(m, n, dtype=torch.float32, device=device) * std + mean
    G = G_raw / (torch.linalg.norm(G_raw, dim=(-2, -1), keepdim=True).clamp_min(1e-8))

    # Generate W (independent random matrix, different from G)
    torch.manual_seed(seed + 1000)  # Use different seed for independence
    W_raw = torch.randn(m, n, dtype=torch.float32, device=device) * std + mean

    # Compute SVD of W to get max singular value and vectors
    U, S, Vh = torch.linalg.svd(W_raw, full_matrices=False)
    max_singular_value = S[0].item()

    # Compute target radius: sqrt(fan_out / fan_in)
    fan_out, fan_in = m, n
    target_radius = (fan_out / fan_in) ** 0.5

    # Normalize W to the spectral ball: W = W_raw * (target_radius / max_singular_value)
    W = W_raw * (target_radius / max_singular_value)

    u, s, vh = torch.linalg.svd(W, full_matrices=False)
    # Get Theta from W's leading singular vectors: Theta = u @ v^T
    u = u[:, :1]  # Leading left singular vector
    v = vh[:1, :].T  # Leading right singular vector
    Theta = u @ v.transpose(-2, -1)

    return G, Theta


# ============================================================================
# Numerical analysis functions
# ============================================================================


def find_zero_point(
    G: torch.Tensor,
    Theta: torch.Tensor,
    lambda_min: float = -1.0,
    lambda_max: float = 1.0,
    msign_steps: int = 8,
) -> Tuple[float, bool]:
    """
    Find the zero point of f(lambda) using Brent's method.

    Args:
        G: Gradient tensor
        Theta: Direction tensor
        lambda_min: Minimum lambda value for search
        lambda_max: Maximum lambda value for search
        msign_steps: Number of msign steps

    Returns:
        zero_point: The lambda value where f(lambda) = 0
        success: Whether the zero point was found
    """

    # Create a wrapper function for scipy
    def f_wrapper(lam):
        return compute_f(G, Theta, lam, msign_steps)

    # Check if there's a sign change in the interval
    f_min = f_wrapper(lambda_min)
    f_max = f_wrapper(lambda_max)

    if f_min * f_max > 0:
        # No sign change, try to expand the search range
        print(
            f"Warning: No sign change in [{lambda_min}, {lambda_max}]. "
            f"f({lambda_min})={f_min:.6e}, f({lambda_max})={f_max:.6e}"
        )

        # Expand range
        if f_min > 0:
            lambda_min *= 10
        else:
            lambda_max *= 10

        f_min = f_wrapper(lambda_min)
        f_max = f_wrapper(lambda_max)

        if f_min * f_max > 0:
            print(
                f"Still no sign change in expanded range [{lambda_min}, {lambda_max}]"
            )
            return 0.0, False

    try:
        zero_point = brentq(f_wrapper, lambda_min, lambda_max, xtol=1e-10)
        return zero_point, True
    except ValueError as e:
        print(f"Error finding zero point: {e}")
        return 0.0, False


def plot_f_lambda_single(
    G: torch.Tensor,
    Theta: torch.Tensor,
    lambda_range: Tuple[float, float] = (-0.1, 0.1),
    num_points: int = 200,
    msign_steps: int = 8,
    title: str = "f(λ) = <Θ, msign(G + λ·Θ)>",
    save_path: str = None,
    show_plot: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Plot f(lambda) and mark the zero point for a single (G, Theta) pair.

    NOTE: This is kept mainly for reference / backward-compatibility.
    The main test loop below now uses a multi-repeat version that overlays
    multiple curves in one figure and computes statistics.

    Args:
        G: Gradient tensor
        Theta: Direction tensor
        lambda_range: Range of lambda values to plot
        num_points: Number of points to sample
        msign_steps: Number of msign steps
        title: Plot title
        save_path: Path to save the plot (optional)
        show_plot: Whether to immediately show this single plot

    Returns:
        lambdas: Array of lambda values
        f_values: Array of f(lambda) values
        zero_point: The lambda value where f(lambda) = 0
    """
    lambda_min, lambda_max = lambda_range
    lambdas = np.linspace(lambda_min, lambda_max, num_points)
    f_values = np.array([compute_f(G, Theta, lam, msign_steps) for lam in lambdas])

    # Find zero point
    zero_point, success = find_zero_point(G, Theta, lambda_min, lambda_max, msign_steps)

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, f_values, "b-", linewidth=2, label="f(λ)")
    plt.axhline(y=0, color="k", linestyle="--", alpha=0.3, label="f(λ) = 0")
    plt.axvline(x=0, color="gray", linestyle="--", alpha=0.3, label="λ = 0")

    if success:
        f_zero = compute_f(G, Theta, zero_point, msign_steps)
        plt.plot(
            zero_point,
            f_zero,
            "ro",
            markersize=10,
            label=f"Zero point: λ={zero_point:.6e}, f={f_zero:.6e}",
        )

    plt.xlabel("λ", fontsize=12)
    plt.ylabel("f(λ)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.tight_layout()

    if show_plot:
        plt.show()

    return lambdas, f_values, zero_point


def plot_f_lambda_multi_repeat(
    m: int,
    n: int,
    mean: float,
    std: float,
    lambda_range: Tuple[float, float] = (-0.1, 0.1),
    num_points: int = 200,
    msign_steps: int = 8,
    n_repeats: int = 5,
    base_seed: int = 42,
    title: str = "f(λ) = <Θ, msign(G + λ·Θ)>",
    save_path: str = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Do n_repeats experiments for a given (m, n, mean, std) config, plot all f(λ)
    curves in one figure, and compute statistics over repeats.

    Args:
        m, n: Matrix size
        mean, std: Initialization config
        lambda_range: Range of lambda values to plot
        num_points: Number of points to sample
        msign_steps: Number of msign steps
        n_repeats: Number of independent repeats
        base_seed: Base random seed; different repeat uses base_seed + repeat_id
        title: Plot title
        save_path: Path to save the plot (optional)
        device: Device to generate matrices and perform computations on (CPU or CUDA)

    Returns:
        lambdas: shape (num_points,)
        f_values_all: shape (n_repeats, num_points)
        f_mean: shape (num_points,)
        f_std: shape (num_points,)
        zero_points: shape (n_repeats,)
    """
    lambda_min, lambda_max = lambda_range
    lambdas = np.linspace(lambda_min, lambda_max, num_points)

    # Store f(λ) for each repeat: (n_repeats, num_points)
    f_values_all = np.zeros((n_repeats, num_points), dtype=np.float64)
    # Store zero point λ* for each repeat
    zero_points = np.zeros(n_repeats, dtype=np.float64)

    plt.figure(figsize=(10, 6))

    for rep in range(n_repeats):
        # 使用不同 seed 进行多次独立试验
        seed = base_seed + rep * 10000
        G, Theta = generate_test_matrices(m, n, mean, std, seed=seed, device=device)

        # 计算这一条曲线的 f(λ)
        f_values = np.array([compute_f(G, Theta, lam, msign_steps) for lam in lambdas])
        f_values_all[rep] = f_values

        # 每条曲线独立找零点
        zero_point, success = find_zero_point(
            G, Theta, lambda_min, lambda_max, msign_steps
        )
        if not success:
            zero_point = np.nan
        zero_points[rep] = zero_point

        # 画出这一条 f(λ) 曲线（所有 repeat 画在同一张图上）
        plt.plot(
            lambdas,
            f_values,
            linewidth=1.5,
            alpha=0.7,
            label=f"repeat {rep+1}" if rep == 0 else None,  # 避免太多 legend 项
        )

    # 对 repeat 维度求 f(λ) 的均值和标准差
    f_mean = np.nanmean(f_values_all, axis=0)
    f_std = np.nanstd(f_values_all, axis=0)

    # λ* 的均值和标准差（忽略 NaN）
    lambda_mean = np.nanmean(zero_points)
    lambda_std = np.nanstd(zero_points)

    # 在图上再画出 mean 曲线，并加一条 ±std 的带状区域
    plt.plot(lambdas, f_mean, "k-", linewidth=2.5, label="mean f(λ) over repeats")
    plt.fill_between(
        lambdas,
        f_mean - f_std,
        f_mean + f_std,
        color="gray",
        alpha=0.2,
        label="±1 std of f(λ)",
    )

    # 画基准线
    plt.axhline(y=0, color="k", linestyle="--", alpha=0.3, label="f(λ) = 0")
    plt.axvline(x=0, color="gray", linestyle="--", alpha=0.3, label="λ = 0")

    # 在图里标出 λ* 的均值位置
    if not np.isnan(lambda_mean):
        # 计算在 λ_mean 处的 f_mean 进行标记（插值）
        f_at_lambda_mean = np.interp(lambda_mean, lambdas, f_mean)
        plt.plot(
            lambda_mean,
            f_at_lambda_mean,
            "ro",
            markersize=8,
            label=f"λ* mean = {lambda_mean:.3e} ± {lambda_std:.1e}",
        )

    plt.xlabel("λ", fontsize=12)
    plt.ylabel("f(λ)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    # 控制外部是否 plt.show()，由 test_f_function 决定
    return lambdas, f_values_all, f_mean, f_std, zero_points


# ============================================================================
# Main test function
# ============================================================================


def test_f_function(
    matrix_sizes: list[Tuple[int, int]] = None,
    init_configs: list[Tuple[float, float]] = None,
    lambda_range: Tuple[float, float] = (-1.0, 1.0),
    msign_steps: int = 8,
    show_plots: bool = True,
    save_plots: bool = False,
    n_repeats: int = 5,  # 新增：每个 size & init config 的重复次数
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """
    Test f(lambda) function for various matrix sizes and initialization configs.

    现在每个 (matrix_size, init_config) 都会做 n_repeats 次独立实验：
    - 5 条 f(λ) 曲线画在同一张图上
    - 统计 λ* 的均值和标准差
    - 统计每个 λ 下 f(λ) 跨 repeat 的均值和标准差

    Args:
        matrix_sizes: List of (m, n) matrix size tuples
        init_configs: List of (mean, std) initialization config tuples
        lambda_range: Range of lambda values to test
        msign_steps: Number of msign steps (default 8 for Polar Express)
        show_plots: Whether to display plots
        save_plots: Whether to save plots to files
        n_repeats: Number of repeats per (size, init) config
        device: Device to generate matrices and perform computations on (CPU or CUDA)
    """
    if matrix_sizes is None:
        matrix_sizes = [
            (1024, 2048),
            (4096, 1024),
            (6144, 1024),
            (1024, 3072),
            (8192, 1024),
            (8, 1024),
        ]

    if init_configs is None:
        init_configs = [
            (0.0, 0.02),  # mean=0.0, std=0.02
            (0.02, 0.02),  # mean=0.02, std=0.06
        ]

    print("=" * 80)
    print(f"Testing f(λ) = <Θ, msign(G + λ·Θ)> with {msign_steps}-step msign")
    print(f"Each (size, init) config is repeated n={n_repeats} times.")
    print("=" * 80)

    results = []

    for m, n in matrix_sizes:
        for mean, std in init_configs:
            print(f"\nMatrix size: {m} x {n}, Init: mean={mean}, std={std}")
            print("-" * 80)

            # =========================
            # repeat n 次并画在同一张图上
            # =========================
            title = f"f(λ) for {m}×{n} matrix (mean={mean}, std={std}), n={n_repeats}"
            save_path = None
            if save_plots:
                save_path = f"f_lambda_{m}x{n}_mean{mean}_std{std}_n{n_repeats}.png"

            lambdas, f_values_all, f_mean, f_std, zero_points = (
                plot_f_lambda_multi_repeat(
                    m,
                    n,
                    mean,
                    std,
                    lambda_range=lambda_range,
                    num_points=200,
                    msign_steps=msign_steps,
                    n_repeats=n_repeats,
                    base_seed=42,
                    title=title,
                    save_path=save_path,
                    device=device,
                )
            )

            # 对每个 repeat 再做一次简单的单点 monotonicity 检查
            # 这里仅在若干固定 λ 上检查，并输出示例
            print(f"Monotonicity check on fixed lambdas (per repeat):")
            test_lambdas = [-1, -0.001, 0.0, 0.001, 1]
            for rep in range(n_repeats):
                seed = 42 + rep * 10000
                G_rep, Theta_rep = generate_test_matrices(
                    m, n, mean, std, seed=seed, device=device
                )
                print(f"  Repeat {rep + 1} (seed={seed}):")
                vals = []
                for lam in test_lambdas:
                    f_val = compute_f(G_rep, Theta_rep, lam, msign_steps)
                    vals.append(f_val)
                    print(f"    f({lam:+.3f}) = {f_val:+.6e}")
                vals = np.array(vals, dtype=np.float64)
                is_monotonic = np.all(np.diff(vals) > 0)
                print(f"    Monotonic over test_lambdas: {is_monotonic}")

            # =========================
            # 输出 lambda* 的均值和标准差
            # =========================
            lambda_mean = np.nanmean(zero_points)
            lambda_std = np.nanstd(zero_points)

            print("\nZero points across repeats:")
            for i, z in enumerate(zero_points):
                print(f"  repeat {i+1}: λ* = {z:.10e}")
            print(f"λ* mean = {lambda_mean:.10e}, std = {lambda_std:.10e}")

            # =========================
            # 输出 f(λ) 均值和标准差的一些信息
            # =========================
            print("\nSample of f(λ) mean/std over repeats (5 λ-points):")
            sample_indices = np.linspace(0, len(lambdas) - 1, num=5, dtype=int)
            for idx in sample_indices:
                lam = lambdas[idx]
                print(
                    f"  λ = {lam:+.3e}: "
                    f"mean f(λ) = {f_mean[idx]:+.6e}, "
                    f"std f(λ) = {f_std[idx]:.6e}"
                )

            results.append(
                {
                    "shape": (m, n),
                    "mean": mean,
                    "std": std,
                    "lambda_mean": lambda_mean,
                    "lambda_std": lambda_std,
                    "zero_points": zero_points.copy(),
                }
            )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Shape':<15} {'Mean':>8} {'Std':>8} {'λ* mean':>15} {'λ* std':>15}")
    print("-" * 80)
    for r in results:
        shape_str = f"{r['shape'][0]}x{r['shape'][1]}"
        print(
            f"{shape_str:<15} "
            f"{r['mean']:>8.3f} "
            f"{r['std']:>8.3f} "
            f"{r['lambda_mean']:>15.6e} "
            f"{r['lambda_std']:>15.6e}"
        )

    print("\n" + "=" * 80)
    all_near_zero = all(
        np.all(np.abs(cfg_result["zero_points"]) < 0.01) for cfg_result in results
    )
    print(f"All zero points near 0 (|λ*| < 1e-2 for all repeats): {all_near_zero}")
    print("=" * 80)

    if show_plots:
        plt.show()


if __name__ == "__main__":
    # Run tests with default configurations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_f_function(
        msign_steps=8,  # Use 8-step Polar Express
        lambda_range=(-1.0, 1.0),
        show_plots=True,
        save_plots=True,
        n_repeats=5,  # 默认做 5 次 repeat
        device=device,
    )
