import torch

# =========================
# msign: Polar-Express 版本
# =========================

def _muon_newton_schulz_step(X: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    """One Newton-Schulz iteration: X ← a·X + X·(b·A + c·A²) where A = X·X^T."""
    A = X @ X.mT
    B = torch.addmm(A, A, A, alpha=c, beta=b)
    X = torch.addmm(X, B, X, alpha=1.0, beta=a)
    return X

@torch.compile
def msign(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Matrix sign via Newton-Schulz with Polar-Express coefficients."""
    if G.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions.")
    if G.dtype != torch.float32:
        raise ValueError("Input tensor G must be in float32")

    transpose_needed = G.size(-2) > G.size(-1)
    X = G.mT if transpose_needed else G
    X = torch.nn.functional.normalize(X, p=2, dim=(-2, -1), eps=1e-7)
    X = X.to(torch.bfloat16)

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

    return X.mT if transpose_needed else X


# =========================
# mclip: mclip-3（单位谱球）
# =========================

@torch.compile
def mclip_unit(W: torch.Tensor, steps: int = 8) -> torch.Tensor:
    r"""
    mclip_{[-1,1]}(W) using Su's mclip-3 formula:

        mclip(W) = 0.5 * [ (S + W) @ ms(W^T W + I)
                         + (S - W) @ ms(W^T W - I) ]

    where S = msign(W), ms = msign.
    """
    W_float32 = W.to(torch.float32)

    # First msign: S = msign(W)
    S = msign(W_float32, steps=steps).to(torch.float32)

    # Gram matrix: G = W^T W (supports batch / non-square matrices)
    G = W_float32.transpose(-2, -1) @ W_float32
    n = G.size(-1)
    I = torch.eye(n, device=W_float32.device, dtype=W_float32.dtype)

    # Two symmetric matrices: G ± I
    G_plus_I  = G + I
    G_minus_I = G - I

    # Compute msign for G±I (Second and third msign calls)
    P = msign(G_plus_I,  steps=steps).to(torch.float32)
    M = msign(G_minus_I, steps=steps).to(torch.float32)

    # Apply the main mclip-3 formula
    output_float32 = 0.5 * ((S + W_float32) @ P + (S - W_float32) @ M)

    return output_float32


# =========================
# SVD Baseline: 精确 mclip
# =========================

@torch.no_grad()
def mclip_unit_svd(W: torch.Tensor) -> torch.Tensor:
    """
    Baseline mclip via SVD:
        W = U Σ V^T
        Σ' = clip(Σ, [0, 1])
        mclip(W) = U Σ' V^T
    """
    W32 = W.to(torch.float32)
    U, S, Vh = torch.linalg.svd(W32, full_matrices=False)  # (..., m, k), (..., k), (..., k, n)
    S_clipped = torch.clamp(S, max=1.0)  # [0, 1]，对奇异值来说 [-1,1] 和 [0,1] 等价

    # 把对角奇异值矩阵嵌回去：U @ diag(S') @ V^T
    Sigma = torch.diag_embed(S_clipped)              # (..., k, k)
    W_clipped = U @ Sigma @ Vh                       # (..., m, n)
    return W_clipped


# =========================
# 误差度量 & Benchmark
# =========================

@torch.no_grad()
def rms_error(A: torch.Tensor, B: torch.Tensor) -> float:
    """
    Root Mean Square error in fp32.
    """
    diff = (A.to(torch.float32) - B.to(torch.float32)).reshape(-1)
    return torch.sqrt(torch.mean(diff * diff)).item()


@torch.no_grad()
def benchmark_mclip(
    shapes = ((128, 128), (256, 256), (1024, 1024), (1024, 4096)),
    steps: int = 8,
    num_mats: int = 3,
    device: str = None,
):
    """
    三连测：
    - approx 结果的谱范数（以及 exact 的谱范数，理论应为 1）
    - 奇异值 MAE（approx vs exact_clipped）
    - 矩阵 RMS 误差（approx vs exact_clipped）
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}, steps={steps}, num_mats={num_mats}")
    torch.manual_seed(0)

    for (m, n) in shapes:
        sn_exact_list = []
        sn_approx_list = []
        mae_sv_list = []
        rms_list = []

        for _ in range(num_mats):
            # 随机矩阵；如果想测极端谱，可以自己改造 W 的构造方式
            W = torch.randn(m, n, device=device, dtype=torch.float32)

            # exact: SVD clip
            exact = mclip_unit_svd(W)         # fp32
            # approx: mclip-3 via msign
            approx = mclip_unit(W, steps=steps)  # 已经是 fp32

            # 1) 谱范数（最大奇异值）
            s_exact = torch.linalg.svdvals(exact)   # 长度 = min(m, n)
            s_approx = torch.linalg.svdvals(approx)

            sn_exact = s_exact[0].item()
            sn_approx = s_approx[0].item()

            sn_exact_list.append(sn_exact)
            sn_approx_list.append(sn_approx)

            # 2) 奇异值 MAE
            mae_sv = torch.mean(torch.abs(s_approx - s_exact)).item()
            mae_sv_list.append(mae_sv)

            # 3) 矩阵 RMS 误差
            rms = rms_error(approx, exact)
            rms_list.append(rms)

        sn_exact_t = torch.tensor(sn_exact_list)
        sn_approx_t = torch.tensor(sn_approx_list)
        mae_sv_t = torch.tensor(mae_sv_list)
        rms_t = torch.tensor(rms_list)

        print(f"\nshape {m}x{n}:")
        print(f"  Spectral norm (exact):  mean = {sn_exact_t.mean().item():.6f}, "
              f"max = {sn_exact_t.max().item():.6f}")
        print(f"  Spectral norm (approx): mean = {sn_approx_t.mean().item():.6f}, "
              f"max = {sn_approx_t.max().item():.6f}")

        print(f"  SV MAE (approx vs exact): mean = {mae_sv_t.mean().item():.3e}, "
              f"max = {mae_sv_t.max().item():.3e}")

        print(f"  Matrix RMS error:        mean = {rms_t.mean().item():.3e}, "
              f"max = {rms_t.max().item():.3e}")


if __name__ == "__main__":
    benchmark_mclip()
