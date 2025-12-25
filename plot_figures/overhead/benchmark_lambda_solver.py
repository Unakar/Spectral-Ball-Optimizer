"""
Benchmark tool for Lambda Solver performance in SpectralBall optimizer.

This is a standalone script that measures:
1. Bisection iteration counts (n_steps)
2. Bracket expansion steps
3. Total solve time vs pure msign baseline
4. Overhead ratio introduced by infra lambda constraint

Results are formatted for paper tables.

Usage:
    python benchmark_lambda_solver.py --quick --trials 5
    python benchmark_lambda_solver.py --compare-compile --quick
    python benchmark_lambda_solver.py --compile --quick
    python benchmark_lambda_solver.py --syrk --quick  # requires Triton >= 3.4.0
"""

import os
import sys
import time
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field
import json
from datetime import datetime

import torch
import numpy as np


# ============================================================================
# Triton SYRK Kernel (optional, requires Triton >= 3.4.0)
# Computes D = alpha * A * A^T + beta * C efficiently
# ============================================================================

HAS_TRITON_SYRK = False
tsyrk_ex = None

try:
    import triton
    import triton.language as tl
    
    # Check for TensorDescriptor (requires Triton >= 3.4.0)
    try:
        from triton.tools.tensor_descriptor import TensorDescriptor
        HAS_TRITON_340 = True
    except ImportError:
        HAS_TRITON_340 = False
    
    if HAS_TRITON_340:
        # ========== SYRK Kernel Implementation ==========
        
        def prune_invalid_configs(configs: list, named_args: dict, **kwargs) -> list:
            """Prune invalid Triton kernel configs based on input size and tile parameters."""
            N = named_args["N"]
            conf = []
            for c in configs:
                TILE_M = c.kwargs.get("TILE_M", 0)
                TILE_N = c.kwargs.get("TILE_N", 0)
                TILE_K = c.kwargs.get("TILE_K", 0)
                if N >= 5000:
                    if TILE_M == 128 and TILE_N == 256 and TILE_K == 64:
                        conf.append(c)
                else:
                    if TILE_M <= 128 and TILE_N >= TILE_M and TILE_K <= 128:
                        conf.append(c)
            return conf

        def matmul_tma_set_block_size_hook(nargs: dict) -> None:
            """Sets the block shapes for tensor descriptors based on tile sizes."""
            TILE_M = nargs["TILE_M"]
            TILE_N = nargs["TILE_N"]
            TILE_K = nargs["TILE_K"]
            TRANS = nargs["TRANS"]
            nargs["a_desc"].block_shape = [TILE_K, TILE_M] if TRANS else [TILE_M, TILE_K]
            nargs["a_t_desc"].block_shape = [TILE_K, TILE_N] if TRANS else [TILE_N, TILE_K]
            if nargs["c_desc"] is not None:
                nargs["c_desc"].block_shape = [TILE_M, TILE_N]
            nargs["d_desc"].block_shape = [TILE_M, TILE_N]
            nargs["d_t_desc"].block_shape = [TILE_N, TILE_M]

        _SYRK_CONFIGS = [
            triton.Config(
                {"TILE_M": tm, "TILE_N": tn, "TILE_K": tk, "GROUP_SIZE_M": gm},
                num_warps=nw,
                num_stages=ns,
                num_ctas=nc,
                pre_hook=matmul_tma_set_block_size_hook,
            )
            for tm in (64, 128, 256)
            for tn in (64, 128, 256)
            for tk in (64, 128, 256)
            for gm in (2, 4, 8)
            for nw in (4, 8)
            for ns in (2, 3, 4)
            for nc in (1,)
        ]

        # Reduce configs for testing
        if "absl.testing" in sys.modules.keys():
            _SYRK_CONFIGS = _SYRK_CONFIGS[:1]

        @triton.autotune(
            configs=_SYRK_CONFIGS,
            key=["N", "K", "TRANS", "WARP_SPECIALIZE"],
            prune_configs_by={"early_config_prune": prune_invalid_configs},
        )
        @triton.jit
        def syrk_kernel_bf16(
            d_desc,
            d_t_desc,
            a_desc,
            a_t_desc,
            c_desc,
            alpha: tl.constexpr,
            beta: tl.constexpr,
            SKIP_UPPER_TRIANGLE: tl.constexpr,
            TRANS: tl.constexpr,
            N: tl.constexpr,
            K: tl.constexpr,
            TILE_M: tl.constexpr,
            TILE_N: tl.constexpr,
            TILE_K: tl.constexpr,
            GROUP_SIZE_M: tl.constexpr,
            WARP_SPECIALIZE: tl.constexpr,
        ):
            """SYRK kernel: D = alpha * A * A^T + beta * C"""
            pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(N, TILE_M)
            num_pid_n = tl.cdiv(N, TILE_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m

            IS_BELOW_DIAG = pid_m * TILE_M >= pid_n * TILE_N + TILE_N
            IS_ABOVE_DIAG = pid_m * TILE_M + TILE_M <= pid_n * TILE_N
            IS_SQUARE_TILE = TILE_M == TILE_N

            if IS_ABOVE_DIAG:
                return

            tl.assume(pid_m >= 0)
            tl.assume(pid_n >= 0)

            offs_row = pid_m * TILE_M
            offs_col = pid_n * TILE_N

            acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

            num_tiles_k = tl.cdiv(K, TILE_K)
            for k in tl.range(num_tiles_k, warp_specialize=WARP_SPECIALIZE):
                offs_k = k * TILE_K
                if TRANS:
                    x = a_desc.load([offs_k, offs_row])
                    y = a_t_desc.load([offs_k, offs_col])
                    acc = tl.dot(x.T, y, acc=acc)
                else:
                    x = a_desc.load([offs_row, offs_k])
                    y = a_t_desc.load([offs_col, offs_k])
                    acc = tl.dot(x, y.T, acc=acc)

            if alpha != 1.0:
                acc = alpha * acc
            if beta != 0.0:
                z = c_desc.load([offs_row, offs_col]).to(tl.float32)
                acc = beta * z + acc

            d = acc.to(tl.bfloat16)

            offs_row = pid_m * TILE_M
            offs_col = pid_n * TILE_N
            d_desc.store([offs_row, offs_col], d)

            if (IS_SQUARE_TILE and IS_BELOW_DIAG) or (not IS_SQUARE_TILE and not IS_ABOVE_DIAG):
                if not SKIP_UPPER_TRIANGLE:
                    d_t_desc.store([offs_col, offs_row], d.T)

        def tsyrk_ex_impl(
            a: torch.Tensor, 
            c: torch.Tensor = None, 
            alpha: float = 1.0, 
            beta: float = 0.0, 
            skip_upper_triangle: bool = False
        ) -> torch.Tensor:
            """Triton implementation of bf16 syrk operation.
            
            Computes: D = alpha * A * A^T + beta * C
            
            Args:
                a: Input tensor of shape (N, K) in bfloat16
                c: None or symmetric input tensor of shape (N, N)
                alpha: Scaling factor for the matrix multiplication
                beta: Scaling factor for the matrix addition
                skip_upper_triangle: Whether to skip the upper triangle part of the output
            
            Returns:
                Output tensor of shape (N, N)
            """
            if a.dtype != torch.bfloat16:
                raise TypeError("Input tensor must be bfloat16")
            if a.dim() != 2:
                raise TypeError("Input tensor must be 2D")
            if not (a.is_contiguous() or a.T.is_contiguous()):
                raise TypeError("invalid input tensor layout. a or a.T must be contiguous.")

            N, K = a.shape
            if not ((c is None and beta == 0.0) or (c is not None and c.shape == (N, N))):
                raise RuntimeError("if c is provided, c must be of shape (N, N)")
            if not (c is None or c.is_contiguous() or c.T.is_contiguous()):
                raise RuntimeError("if c is provided, c or c.T must be contiguous")

            d = torch.empty((N, N), device=a.device, dtype=a.dtype)

            dummy_block = [1, 1]

            is_trans = a.T.is_contiguous()

            if is_trans:
                a = a.T
            a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
            a_t_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
            d_desc = TensorDescriptor(d, d.shape, d.stride(), dummy_block)
            d_t_desc = TensorDescriptor(d, d.shape, d.stride(), dummy_block)

            if beta != 0.0:
                c = c.T if c.T.is_contiguous() else c
                c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)
            else:
                c_desc = None

            def grid(META):
                return (triton.cdiv(N, META["TILE_M"]) * triton.cdiv(N, META["TILE_N"]),)

            syrk_kernel_bf16[grid](
                d_desc,
                d_t_desc,
                a_desc,
                a_t_desc,
                c_desc,
                alpha,
                beta,
                skip_upper_triangle,
                is_trans,
                N,
                K,
                WARP_SPECIALIZE=False,
            )
            return d
        
        # Export the function
        tsyrk_ex = tsyrk_ex_impl
        HAS_TRITON_SYRK = True
        print("✅ Triton SYRK kernel available (Triton >= 3.4.0)")
    else:
        print("⚠️ Triton SYRK kernel requires Triton >= 3.4.0")
        
except ImportError as e:
    print(f"⚠️ Triton not available: {e}")
except Exception as e:
    print(f"⚠️ Triton SYRK kernel initialization failed: {e}")


# ============================================================================
# Global flags for optimizations
# ============================================================================

USE_TORCH_COMPILE = False  # Will be set by command line arg
USE_TRITON_SYRK = False    # Will be set by command line arg


def set_use_compile(use_compile: bool):
    """Set global flag for torch.compile usage."""
    global USE_TORCH_COMPILE
    USE_TORCH_COMPILE = use_compile


def set_use_triton_syrk(use_syrk: bool):
    """Set global flag for Triton SYRK kernel usage."""
    global USE_TRITON_SYRK
    if use_syrk and not HAS_TRITON_SYRK:
        print("⚠️ Triton SYRK requested but not available, falling back to torch.matmul")
        USE_TRITON_SYRK = False
    else:
        USE_TRITON_SYRK = use_syrk


# ============================================================================
# Core functions: Newton-Schulz iterations for matrix sign
# ============================================================================

def _muon_newton_schulz_step(X: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    """One Newton-Schulz iteration: X ← a·X + X·(b·A + c·A²) where A = X·X^T."""
    A = X @ X.mT
    B = torch.addmm(A, A, A, alpha=c, beta=b)
    X = torch.addmm(X, B, X, alpha=1.0, beta=a)
    return X


def _muon_newton_schulz_step_syrk(X: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    """One Newton-Schulz iteration using Triton SYRK kernel for A = X @ X^T.
    
    X ← a·X + X·(b·A + c·A²) where A = X·X^T
    
    The SYRK kernel computes D = alpha * A * A^T + beta * C, which we use for:
    1. A = X @ X^T (using tsyrk_ex with alpha=1, beta=0)
    """
    # Use Triton SYRK kernel for A = X @ X^T
    # Note: tsyrk_ex expects bfloat16 input and X is already bfloat16 in msign
    A = tsyrk_ex(X, alpha=1.0, beta=0.0)
    
    # B = b*A + c*A*A  (A*A is also a symmetric matrix product, but A is NxN)
    # For now, use torch for this part since A is smaller (NxN where N = min(m,n))
    B = torch.addmm(A, A, A, alpha=c, beta=b)
    
    # X = a*X + B @ X
    X = torch.addmm(X, B, X, alpha=1.0, beta=a)
    return X


# Polar-Express coefficients for Newton-Schulz iterations
POLAR_EXPRESS_COEFFS = [
    (8.2051, -22.9019, 16.4607),
    (4.0664, -2.8612, 0.5184),
    (3.9096, -2.8234, 0.5250),
    (3.2856, -2.4153, 0.4853),
    (2.2779, -1.6198, 0.3985),
    (1.8726, -1.2307, 0.3585),
    (1.8564, -1.2132, 0.3568),
    (1.8750, -1.2500, 0.3750),
]


def _msign_impl(G: torch.Tensor, steps: int, use_syrk: bool = False) -> torch.Tensor:
    """Matrix sign via Newton-Schulz with Polar-Express coefficients (implementation).
    
    Args:
        G: Input tensor of shape (m, n) in float32
        steps: Number of Newton-Schulz iterations
        use_syrk: Whether to use Triton SYRK kernel for A = X @ X^T
    """
    if G.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions.")
    if G.dtype != torch.float32:
        raise ValueError(f"Input tensor G must be in float32")

    transpose = G.size(-2) > G.size(-1)
    X = G.mT if transpose else G
    X = torch.nn.functional.normalize(X, p=2, dim=(-2, -1), eps=1e-7)
    X = X.to(torch.bfloat16)

    # Select step function based on whether SYRK is enabled
    step_fn = _muon_newton_schulz_step_syrk if (use_syrk and HAS_TRITON_SYRK) else _muon_newton_schulz_step

    for i in range(steps):
        if i < 8:
            a, b, c = POLAR_EXPRESS_COEFFS[i]
        else:
            a, b, c = POLAR_EXPRESS_COEFFS[-1]
        X = step_fn(X, a, b, c)

    return X.mT if transpose else X


# Compiled version of msign (without SYRK, since Triton kernels handle their own compilation)
_msign_compiled = None


def get_msign_compiled():
    """Lazily compile msign function (non-SYRK version)."""
    global _msign_compiled
    if _msign_compiled is None:
        # Compile the non-SYRK version
        def msign_no_syrk(G, steps):
            return _msign_impl(G, steps, use_syrk=False)
        _msign_compiled = torch.compile(msign_no_syrk)
    return _msign_compiled


def msign(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Matrix sign - dispatches based on optimization flags.
    
    Priority:
    1. If USE_TRITON_SYRK: use SYRK kernel (already optimized, no torch.compile needed)
    2. If USE_TORCH_COMPILE: use torch.compile
    3. Otherwise: use plain implementation
    """
    if USE_TRITON_SYRK and HAS_TRITON_SYRK:
        # SYRK kernel is already highly optimized, torch.compile not needed
        return _msign_impl(G, steps, use_syrk=True)
    elif USE_TORCH_COMPILE:
        return get_msign_compiled()(G, steps)
    else:
        return _msign_impl(G, steps, use_syrk=False)


# ============================================================================
# Power Iteration for leading singular triplet
# ============================================================================

def _power_iteration_impl(w: torch.Tensor, steps: int = 50) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Leading singular triplet (σ, u, v) via bilateral power iteration (fp32) - implementation."""
    if w.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions.")

    w = w.to(torch.float32)
    v = torch.ones_like(w[..., :1, :].transpose(-2, -1))
    for _ in range(steps):
        v = torch.nn.functional.normalize(w.transpose(-2, -1) @ (w @ v), dim=-2)
    u = torch.nn.functional.normalize(w @ v, dim=-2)
    s = (u.transpose(-2, -1) @ w @ v).squeeze(-1).squeeze(-1)

    return s, u, v


# Compiled version of power_iteration
_power_iteration_compiled = None


def get_power_iteration_compiled():
    """Lazily compile power_iteration function."""
    global _power_iteration_compiled
    if _power_iteration_compiled is None:
        _power_iteration_compiled = torch.compile(_power_iteration_impl)
    return _power_iteration_compiled


@torch.no_grad()
def power_iteration(w: torch.Tensor, steps: int = 50, eps: float = 1e-20):
    """Power iteration - dispatches to compiled or non-compiled version."""
    if USE_TORCH_COMPILE:
        return get_power_iteration_compiled()(w, steps)
    else:
        return _power_iteration_impl(w, steps)


# ============================================================================
# Lambda solver helper functions
# ============================================================================

@torch.no_grad()
def inner_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Frobenius inner product <a, b>."""
    return (a * b).sum()


@torch.no_grad()
def compute_phi(G: torch.Tensor, Theta: torch.Tensor, lambda_value: float, msign_steps: int = 8) -> torch.Tensor:
    """Φ(λ) = msign(G + λΘ)."""
    z = G + lambda_value * Theta
    Phi = msign(z, steps=msign_steps)
    return Phi


@torch.no_grad()
def compute_f(G: torch.Tensor, Theta: torch.Tensor, lambda_value: float, msign_steps: int = 8) -> float:
    """f(λ) = <Θ, msign(G + λΘ)>."""
    Phi = compute_phi(G, Theta, lambda_value, msign_steps)
    f_value = float(inner_product(Theta, Phi).item())
    return f_value


# ============================================================================
# Instrumented solver functions (return detailed profiling info)
# ============================================================================

@dataclass
class SolverStats:
    """Statistics from one solver run."""
    lambda_value: float = 0.0
    converged: bool = False
    residual: float = 0.0
    bisection_steps: int = 0
    bracket_steps: int = 0
    total_msign_calls: int = 0
    solve_time_ms: float = 0.0
    bracket_time_ms: float = 0.0
    bisection_time_ms: float = 0.0


@torch.no_grad()
def find_bracket_instrumented(
    G: torch.Tensor,
    Theta: torch.Tensor,
    initial_guess: float = 0.0,
    initial_step: float = 1e-3,
    max_expansions: int = 10,
    msign_steps: int = 8,
    tolerance_f: float = 1e-8,
) -> Tuple[Optional[float], Optional[float], float, float, int]:
    """
    Instrumented find_bracket that returns step count.
    
    Returns:
        (λ_L, λ_R, f_L, f_R, bracket_steps)
    """
    f = compute_f
    bracket_steps = 0

    λ0 = initial_guess
    f0 = f(G, Theta, λ0, msign_steps)
    bracket_steps += 1

    if abs(f0) < tolerance_f:
        return λ0, λ0, f0, f0, bracket_steps

    step = initial_step if f0 < 0 else -initial_step

    λ_prev = λ0
    f_prev = f0

    for _ in range(max_expansions):
        λ_new = λ_prev + step
        f_new = f(G, Theta, λ_new, msign_steps)
        bracket_steps += 1

        sign_prev = f_prev <= 0.0
        sign_new = f_new <= 0.0

        if sign_prev != sign_new:
            if f_prev <= 0 and f_new >= 0:
                λ_L, f_L = λ_prev, f_prev
                λ_R, f_R = λ_new, f_new
            elif f_new <= 0 and f_prev >= 0:
                λ_L, f_L = λ_new, f_new
                λ_R, f_R = λ_prev, f_prev
            else:
                if abs(f_prev) <= abs(f_new):
                    λ_L = λ_R = λ_prev
                    f_L = f_R = f_prev
                else:
                    λ_L = λ_R = λ_new
                    f_L = f_R = f_new

            return λ_L, λ_R, f_L, f_R, bracket_steps

        step *= 2.0
        λ_prev, f_prev = λ_new, f_new

    return None, None, f0, f0, bracket_steps


@torch.no_grad()
def solve_lambda_instrumented(
    G: torch.Tensor,
    Theta: torch.Tensor,
    initial_guess: float = 0.0,
    initial_step: float = 1e-3,
    tolerance_f: float = 1e-6,
    max_iterations: int = 20,
    max_expansions: int = 10,
    msign_steps: int = 8,
    device: str = "cuda",
) -> SolverStats:
    """
    Instrumented bisection solver that returns detailed stats.
    """
    stats = SolverStats()
    
    # Ensure GPU sync for accurate timing
    if device == "cuda":
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    bracket_start = time.perf_counter()
    
    # Find bracket
    λ_L, λ_R, f_L, f_R, bracket_steps = find_bracket_instrumented(
        G, Theta,
        initial_guess=initial_guess,
        initial_step=initial_step,
        max_expansions=max_expansions,
        msign_steps=msign_steps,
        tolerance_f=tolerance_f,
    )
    
    if device == "cuda":
        torch.cuda.synchronize()
    bracket_end = time.perf_counter()
    stats.bracket_time_ms = (bracket_end - bracket_start) * 1000
    stats.bracket_steps = bracket_steps

    # Bracketing failed
    if λ_L is None:
        stats.residual = f_L
        stats.solve_time_ms = (time.perf_counter() - start_time) * 1000
        stats.total_msign_calls = bracket_steps
        return stats

    # Pick best endpoint
    if abs(f_L) < abs(f_R):
        best_λ, best_f = λ_L, f_L
    else:
        best_λ, best_f = λ_R, f_R

    if abs(best_f) <= tolerance_f:
        stats.lambda_value = best_λ
        stats.converged = True
        stats.residual = abs(best_f)
        stats.bisection_steps = 0
        if device == "cuda":
            torch.cuda.synchronize()
        stats.solve_time_ms = (time.perf_counter() - start_time) * 1000
        stats.total_msign_calls = bracket_steps
        return stats

    # Bisection
    bisection_start = time.perf_counter()
    bisection_steps = 0
    
    for it in range(1, max_iterations + 1):
        λ_mid = 0.5 * (λ_L + λ_R)
        f_mid = compute_f(G, Theta, λ_mid, msign_steps)
        bisection_steps += 1

        if abs(f_mid) < abs(best_f):
            best_λ, best_f = λ_mid, f_mid

        if abs(f_mid) <= tolerance_f:
            stats.lambda_value = λ_mid
            stats.converged = True
            stats.residual = abs(f_mid)
            stats.bisection_steps = bisection_steps
            break

        if f_mid < 0:
            λ_L, f_L = λ_mid, f_mid
        else:
            λ_R, f_R = λ_mid, f_mid
    else:
        stats.lambda_value = best_λ
        stats.converged = False
        stats.residual = abs(best_f)
        stats.bisection_steps = bisection_steps

    if device == "cuda":
        torch.cuda.synchronize()
    bisection_end = time.perf_counter()
    
    stats.bisection_time_ms = (bisection_end - bisection_start) * 1000
    stats.solve_time_ms = (time.perf_counter() - start_time) * 1000
    stats.total_msign_calls = bracket_steps + bisection_steps
    
    return stats


# ============================================================================
# Test matrix generation
# ============================================================================

def generate_test_matrices(
    m: int,
    n: int,
    mean: float = 0.0,
    std: float = 0.02,
    seed: int = 42,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate test matrices G, Theta, and W.
    
    Returns:
        G: Normalized gradient tensor
        Theta: Rank-1 direction from SVD
        W: Weight matrix (for computing target radius)
    """
    torch.manual_seed(seed)
    
    # Generate G (normalized gradient)
    G_raw = torch.randn(m, n, dtype=torch.float32, device=device) * std + mean
    G = G_raw / (torch.linalg.norm(G_raw, dim=(-2, -1), keepdim=True).clamp_min(1e-8))

    # Generate W (weight matrix)
    torch.manual_seed(seed + 1000)
    W_raw = torch.randn(m, n, dtype=torch.float32, device=device) * std + mean

    # SVD for Theta
    U, S, Vh = torch.linalg.svd(W_raw, full_matrices=False)
    max_singular_value = S[0].item()
    
    # Normalize W to spectral ball
    fan_out, fan_in = m, n
    target_radius = (fan_out / fan_in) ** 0.5
    W = W_raw * (target_radius / max_singular_value)

    # Get Theta from leading singular vectors
    u = U[:, :1]
    v = Vh[:1, :].T
    Theta = u @ v.transpose(-2, -1)

    return G, Theta, W


# ============================================================================
# Benchmark functions
# ============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark."""
    matrix_shapes: List[Tuple[int, int]] = field(default_factory=lambda: [
        # Common Transformer shapes (GPT-like)
        (1024, 4096),   # FFN up projection
        (4096, 1024),   # FFN down projection
        (1024, 1024),   # Attention QKV (square)
        (6144, 1024),   # Wide FFN
        (1024, 6144),   # Wide FFN (transposed)
        (8192, 1024),   # Very wide
        (1024, 8192),   # Very wide (transposed)
        # Smaller shapes for quick test
        (512, 512),
        (256, 1024),
        (2048, 2048),
    ])
    
    msign_steps_list: List[int] = field(default_factory=lambda: [8])
    tolerances: List[float] = field(default_factory=lambda: [1e-2, 1e-3, 1e-4])
    num_trials: int = 10
    warmup_trials: int = 3
    device: str = "cuda"
    seed: int = 42


@dataclass
class BenchmarkResult:
    """Result for one benchmark configuration."""
    shape: Tuple[int, int]
    msign_steps: int
    tolerance: float
    use_compile: bool = False  # Whether torch.compile was used
    
    # Averaged stats
    avg_bisection_steps: float = 0.0
    std_bisection_steps: float = 0.0
    avg_bracket_steps: float = 0.0
    std_bracket_steps: float = 0.0
    avg_total_msign_calls: float = 0.0
    
    # Time stats (ms)
    avg_solve_time_ms: float = 0.0
    std_solve_time_ms: float = 0.0
    avg_bracket_time_ms: float = 0.0
    avg_bisection_time_ms: float = 0.0
    
    # Baseline comparison (pure msign time)
    avg_pure_msign_time_ms: float = 0.0
    overhead_ratio: float = 0.0  # solve_time / pure_msign_time
    
    # Power iteration benchmark
    avg_power_iter_time_ms: float = 0.0
    
    # Convergence
    convergence_rate: float = 1.0
    
    # Individual trial data
    all_bisection_steps: List[int] = field(default_factory=list)
    all_bracket_steps: List[int] = field(default_factory=list)
    all_solve_times_ms: List[float] = field(default_factory=list)


def benchmark_pure_msign(
    G: torch.Tensor,
    msign_steps: int,
    num_trials: int,
    warmup_trials: int,
    device: str = "cuda"
) -> float:
    """Benchmark pure msign call (baseline)."""
    # Warmup
    for _ in range(warmup_trials):
        _ = msign(G, steps=msign_steps)
        if device == "cuda":
            torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_trials):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = msign(G, steps=msign_steps)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    return np.mean(times)


def benchmark_power_iteration(
    W: torch.Tensor,
    power_iter_steps: int,
    num_trials: int,
    warmup_trials: int,
    device: str = "cuda"
) -> float:
    """Benchmark power_iteration call."""
    # Warmup
    for _ in range(warmup_trials):
        _ = power_iteration(W, steps=power_iter_steps)
        if device == "cuda":
            torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_trials):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = power_iteration(W, steps=power_iter_steps)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    return np.mean(times)


def run_compile_comparison(
    shapes: List[Tuple[int, int]],
    msign_steps: int = 8,
    power_iter_steps: int = 50,
    num_trials: int = 10,
    warmup_trials: int = 5,
    device: str = "cuda",
    include_syrk: bool = True,
) -> Dict:
    """
    Run comparison between different optimization strategies.
    
    Compares:
    1. No optimization (baseline)
    2. torch.compile
    3. Triton SYRK kernel (if available)
    
    Returns a dict with detailed comparison results.
    """
    print("\n" + "=" * 100)
    print("🔥 Optimization Speedup Comparison")
    print("=" * 100)
    
    results = {
        "msign": [],
        "power_iteration": [],
    }
    
    for shape in shapes:
        m, n = shape
        print(f"\n{'='*60}")
        print(f"Matrix shape: {m} × {n}")
        print(f"{'='*60}")
        
        # Generate test tensors
        G, Theta, W = generate_test_matrices(m, n, seed=42, device=device)
        
        # ============== msign comparison ==============
        print(f"\n📊 msign (steps={msign_steps}):")
        
        msign_result = {"shape": shape}
        
        # 1. Baseline: No optimization
        set_use_compile(False)
        set_use_triton_syrk(False)
        msign_baseline = benchmark_pure_msign(G, msign_steps, num_trials, warmup_trials, device)
        print(f"  Baseline:      {msign_baseline:.3f} ms")
        msign_result["baseline_ms"] = msign_baseline
        
        # 2. torch.compile
        set_use_compile(True)
        set_use_triton_syrk(False)
        # Extra warmup for compilation
        for _ in range(5):
            _ = msign(G, steps=msign_steps)
            if device == "cuda":
                torch.cuda.synchronize()
        msign_compiled = benchmark_pure_msign(G, msign_steps, num_trials, warmup_trials, device)
        speedup_compile = msign_baseline / msign_compiled if msign_compiled > 0 else 0
        print(f"  torch.compile: {msign_compiled:.3f} ms (speedup: {speedup_compile:.2f}×)")
        msign_result["compiled_ms"] = msign_compiled
        msign_result["compile_speedup"] = speedup_compile
        
        # 3. Triton SYRK kernel (if available)
        if include_syrk and HAS_TRITON_SYRK:
            set_use_compile(False)
            set_use_triton_syrk(True)
            # Extra warmup for Triton autotuning
            for _ in range(5):
                _ = msign(G, steps=msign_steps)
                if device == "cuda":
                    torch.cuda.synchronize()
            msign_syrk = benchmark_pure_msign(G, msign_steps, num_trials, warmup_trials, device)
            speedup_syrk = msign_baseline / msign_syrk if msign_syrk > 0 else 0
            print(f"  Triton SYRK:   {msign_syrk:.3f} ms (speedup: {speedup_syrk:.2f}×)")
            msign_result["syrk_ms"] = msign_syrk
            msign_result["syrk_speedup"] = speedup_syrk
        else:
            msign_result["syrk_ms"] = None
            msign_result["syrk_speedup"] = None
            if include_syrk:
                print(f"  Triton SYRK:   N/A (requires Triton >= 3.4.0)")
        
        results["msign"].append(msign_result)
        
        # ============== power_iteration comparison ==============
        print(f"\n📊 power_iteration (steps={power_iter_steps}):")
        
        pi_result = {"shape": shape}
        
        # Reset flags
        set_use_compile(False)
        set_use_triton_syrk(False)
        
        # 1. Baseline
        pi_baseline = benchmark_power_iteration(W, power_iter_steps, num_trials, warmup_trials, device)
        print(f"  Baseline:      {pi_baseline:.3f} ms")
        pi_result["baseline_ms"] = pi_baseline
        
        # 2. torch.compile
        set_use_compile(True)
        for _ in range(5):
            _ = power_iteration(W, steps=power_iter_steps)
            if device == "cuda":
                torch.cuda.synchronize()
        pi_compiled = benchmark_power_iteration(W, power_iter_steps, num_trials, warmup_trials, device)
        speedup_pi = pi_baseline / pi_compiled if pi_compiled > 0 else 0
        print(f"  torch.compile: {pi_compiled:.3f} ms (speedup: {speedup_pi:.2f}×)")
        pi_result["compiled_ms"] = pi_compiled
        pi_result["compile_speedup"] = speedup_pi
        
        results["power_iteration"].append(pi_result)
    
    # Reset flags
    set_use_compile(False)
    set_use_triton_syrk(False)
    
    # Print summary table
    print("\n" + "=" * 100)
    print("📋 Summary: Optimization Speedup")
    print("=" * 100)
    
    print("\n### msign ###")
    header = f"{'Shape':<20} {'Baseline (ms)':<15} {'Compile (ms)':<15} {'Compile ↑':<12}"
    if HAS_TRITON_SYRK:
        header += f"{'SYRK (ms)':<15} {'SYRK ↑':<12}"
    print(header)
    print("-" * (90 if HAS_TRITON_SYRK else 65))
    
    for r in results["msign"]:
        shape_str = f"{r['shape'][0]}×{r['shape'][1]}"
        line = f"{shape_str:<20} {r['baseline_ms']:<15.3f} {r['compiled_ms']:<15.3f} {r['compile_speedup']:<12.2f}×"
        if HAS_TRITON_SYRK and r.get('syrk_ms') is not None:
            line += f"{r['syrk_ms']:<15.3f} {r['syrk_speedup']:<12.2f}×"
        print(line)
    
    print("\n### power_iteration ###")
    print(f"{'Shape':<20} {'Baseline (ms)':<15} {'Compile (ms)':<15} {'Compile ↑':<12}")
    print("-" * 65)
    for r in results["power_iteration"]:
        shape_str = f"{r['shape'][0]}×{r['shape'][1]}"
        print(f"{shape_str:<20} {r['baseline_ms']:<15.3f} {r['compiled_ms']:<15.3f} {r['compile_speedup']:<12.2f}×")
    
    return results


def run_benchmark(config: BenchmarkConfig) -> List[BenchmarkResult]:
    """Run the full benchmark suite."""
    results = []
    
    print("=" * 100)
    print("Lambda Solver Benchmark")
    print(f"Device: {config.device}")
    print(f"Trials: {config.num_trials} (+ {config.warmup_trials} warmup)")
    print("=" * 100)
    
    total_configs = len(config.matrix_shapes) * len(config.msign_steps_list) * len(config.tolerances)
    config_idx = 0
    
    for shape in config.matrix_shapes:
        m, n = shape
        print(f"\n{'='*60}")
        print(f"Matrix shape: {m} × {n}")
        print(f"{'='*60}")
        
        for msign_steps in config.msign_steps_list:
            for tolerance in config.tolerances:
                config_idx += 1
                print(f"\n[{config_idx}/{total_configs}] msign_steps={msign_steps}, tol={tolerance:.0e}")
                
                result = BenchmarkResult(
                    shape=shape,
                    msign_steps=msign_steps,
                    tolerance=tolerance,
                )
                
                converged_count = 0
                
                # Run trials with different seeds
                for trial in range(config.warmup_trials + config.num_trials):
                    seed = config.seed + trial * 100
                    
                    # Generate test matrices
                    G, Theta, W = generate_test_matrices(
                        m, n, seed=seed, device=config.device
                    )
                    
                    # Run solver
                    stats = solve_lambda_instrumented(
                        G=G,
                        Theta=Theta,
                        tolerance_f=tolerance,
                        msign_steps=msign_steps,
                        device=config.device,
                    )
                    
                    # Skip warmup trials for statistics
                    if trial >= config.warmup_trials:
                        result.all_bisection_steps.append(stats.bisection_steps)
                        result.all_bracket_steps.append(stats.bracket_steps)
                        result.all_solve_times_ms.append(stats.solve_time_ms)
                        if stats.converged:
                            converged_count += 1
                
                # Compute averages
                result.avg_bisection_steps = np.mean(result.all_bisection_steps)
                result.std_bisection_steps = np.std(result.all_bisection_steps)
                result.avg_bracket_steps = np.mean(result.all_bracket_steps)
                result.std_bracket_steps = np.std(result.all_bracket_steps)
                result.avg_total_msign_calls = result.avg_bisection_steps + result.avg_bracket_steps
                
                result.avg_solve_time_ms = np.mean(result.all_solve_times_ms)
                result.std_solve_time_ms = np.std(result.all_solve_times_ms)
                
                result.convergence_rate = converged_count / config.num_trials
                
                # Get baseline (pure msign)
                G, Theta, W = generate_test_matrices(m, n, seed=config.seed, device=config.device)
                result.avg_pure_msign_time_ms = benchmark_pure_msign(
                    G, msign_steps, config.num_trials, config.warmup_trials, config.device
                )
                
                # Compute overhead ratio
                if result.avg_pure_msign_time_ms > 0:
                    result.overhead_ratio = result.avg_solve_time_ms / result.avg_pure_msign_time_ms
                
                results.append(result)
                
                # Print summary
                print(f"  Bracket steps: {result.avg_bracket_steps:.1f} ± {result.std_bracket_steps:.1f}")
                print(f"  Bisection steps: {result.avg_bisection_steps:.1f} ± {result.std_bisection_steps:.1f}")
                print(f"  Total msign calls: {result.avg_total_msign_calls:.1f}")
                print(f"  Solve time: {result.avg_solve_time_ms:.3f} ± {result.std_solve_time_ms:.3f} ms")
                print(f"  Pure msign: {result.avg_pure_msign_time_ms:.3f} ms")
                print(f"  Overhead ratio: {result.overhead_ratio:.2f}x")
                print(f"  Convergence: {result.convergence_rate*100:.0f}%")
    
    return results


# ============================================================================
# Output formatting (for paper)
# ============================================================================

def format_latex_table(results: List[BenchmarkResult], tolerance: float = 1e-6) -> str:
    """Format results as LaTeX table for paper."""
    # Filter by tolerance
    filtered = [r for r in results if r.tolerance == tolerance]
    
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Lambda Solver Overhead (tol=$" + f"{tolerance:.0e}" + r"$)}",
        r"\label{tab:lambda_solver_overhead}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Shape & msign & Bracket & Bisect & Total msign & Overhead \\",
        r"\midrule",
    ]
    
    for r in filtered:
        shape_str = f"${r.shape[0]}\\times{r.shape[1]}$"
        lines.append(
            f"{shape_str} & {r.msign_steps} & "
            f"{r.avg_bracket_steps:.1f} & {r.avg_bisection_steps:.1f} & "
            f"{r.avg_total_msign_calls:.1f} & {r.overhead_ratio:.1f}$\\times$ \\\\"
        )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def format_markdown_table(results: List[BenchmarkResult]) -> str:
    """Format results as Markdown table."""
    lines = [
        "| Shape | msign | Tolerance | Bracket | Bisect | Total | Time (ms) | Overhead |",
        "|-------|-------|-----------|---------|--------|-------|-----------|----------|",
    ]
    
    for r in results:
        shape_str = f"{r.shape[0]}×{r.shape[1]}"
        tol_str = f"{r.tolerance:.0e}"
        lines.append(
            f"| {shape_str} | {r.msign_steps} | {tol_str} | "
            f"{r.avg_bracket_steps:.1f} | {r.avg_bisection_steps:.1f} | "
            f"{r.avg_total_msign_calls:.1f} | {r.avg_solve_time_ms:.3f} | "
            f"{r.overhead_ratio:.2f}× |"
        )
    
    return "\n".join(lines)


def save_results(results: List[BenchmarkResult], output_dir: str):
    """Save benchmark results to files."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as JSON
    json_data = []
    for r in results:
        json_data.append({
            "shape": list(r.shape),
            "msign_steps": r.msign_steps,
            "tolerance": r.tolerance,
            "avg_bisection_steps": r.avg_bisection_steps,
            "std_bisection_steps": r.std_bisection_steps,
            "avg_bracket_steps": r.avg_bracket_steps,
            "std_bracket_steps": r.std_bracket_steps,
            "avg_total_msign_calls": r.avg_total_msign_calls,
            "avg_solve_time_ms": r.avg_solve_time_ms,
            "std_solve_time_ms": r.std_solve_time_ms,
            "avg_pure_msign_time_ms": r.avg_pure_msign_time_ms,
            "overhead_ratio": r.overhead_ratio,
            "convergence_rate": r.convergence_rate,
        })
    
    json_path = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nSaved JSON results to: {json_path}")
    
    # Save Markdown
    md_path = os.path.join(output_dir, f"results_{timestamp}.md")
    with open(md_path, "w") as f:
        f.write("# Lambda Solver Benchmark Results\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(format_markdown_table(results))
    print(f"Saved Markdown results to: {md_path}")
    
    # Save LaTeX
    for tol in [1e-2, 1e-3, 1e-4]:
        latex_path = os.path.join(output_dir, f"table_tol{tol:.0e}_{timestamp}.tex")
        with open(latex_path, "w") as f:
            f.write(format_latex_table(results, tol))
        print(f"Saved LaTeX table to: {latex_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run benchmark with default or custom configuration."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Benchmark Lambda Solver for SpectralBall optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_lambda_solver.py --quick --trials 5
  python benchmark_lambda_solver.py --compare-compile --quick
  python benchmark_lambda_solver.py --compile --quick
  python benchmark_lambda_solver.py --syrk --quick  # requires Triton >= 3.4.0
        """
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--trials", type=int, default=10, help="Number of trials per config")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup trials")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer shapes")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for msign and power_iteration")
    parser.add_argument("--syrk", action="store_true", help="Use Triton SYRK kernel for msign (requires Triton >= 3.4.0)")
    parser.add_argument("--compare-compile", action="store_true", 
                        help="Compare all optimization strategies: baseline vs compile vs SYRK")
    parser.add_argument("--output-dir", type=str, 
                        default="./results",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    if args.device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Set optimization flags
    if args.syrk:
        set_use_triton_syrk(True)
        if HAS_TRITON_SYRK:
            print("🚀 Triton SYRK kernel ENABLED")
        else:
            print("⚠️ Triton SYRK requested but not available")
    else:
        set_use_triton_syrk(False)
    
    if args.compile and not args.syrk:
        set_use_compile(True)
        print("🔥 torch.compile ENABLED")
    else:
        set_use_compile(False)
        if not args.syrk:
            print("📊 No optimization (use --compile or --syrk to enable)")
    
    # Configure benchmark shapes
    if args.quick:
        shapes = [
            (1024, 4096),
            (4096, 1024),
            (2048, 2048),
        ]
        msign_steps = [8]
        tolerances = [1e-6]
    else:
        shapes = [
            (1024, 4096),
            (4096, 1024),
            (1024, 1024),
            (6144, 1024),
            (1024, 6144),
            (8192, 1024),
            (1024, 8192),
            (2048, 2048),
            (4096, 4096),
        ]
        msign_steps = [8]
        tolerances = [1e-2, 1e-3, 1e-4]
    
    # Run compile comparison if requested
    if args.compare_compile:
        compile_results = run_compile_comparison(
            shapes=shapes,
            msign_steps=8,
            power_iter_steps=50,
            num_trials=args.trials,
            warmup_trials=max(args.warmup, 5),  # Need more warmup for compile
            device=args.device,
        )
        
        # Save compile comparison results
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(args.output_dir, f"compile_comparison_{timestamp}.json")
        with open(json_path, "w") as f:
            # Convert tuples to lists for JSON serialization
            json_data = {
                "msign": [{"shape": list(r["shape"]), **{k: v for k, v in r.items() if k != "shape"}} 
                          for r in compile_results["msign"]],
                "power_iteration": [{"shape": list(r["shape"]), **{k: v for k, v in r.items() if k != "shape"}} 
                                    for r in compile_results["power_iteration"]],
            }
            json.dump(json_data, f, indent=2)
        print(f"\nSaved compile comparison to: {json_path}")
        
        print("\n" + "=" * 100)
        print("Compile comparison complete!")
        print("=" * 100)
        return
    
    config = BenchmarkConfig(
        matrix_shapes=shapes,
        msign_steps_list=msign_steps,
        tolerances=tolerances,
        num_trials=args.trials,
        warmup_trials=args.warmup,
        device=args.device,
    )
    
    # Run benchmark
    results = run_benchmark(config)
    
    # Print summary tables
    print("\n" + "=" * 100)
    print("SUMMARY TABLES")
    print("=" * 100)
    
    print("\n### Markdown Table ###\n")
    print(format_markdown_table(results))
    
    print("\n### LaTeX Table (tol=1e-4) ###\n")
    print(format_latex_table(results, 1e-4))
    
    # Save results
    save_results(results, args.output_dir)
    
    print("\n" + "=" * 100)
    print("Benchmark complete!")
    print("=" * 100)


if __name__ == "__main__":
    main()
