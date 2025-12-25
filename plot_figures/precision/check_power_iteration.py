import torch

@torch.no_grad()
def power_iteration(w: torch.Tensor, steps: int = 50, eps: float = 1e-20):
    """Leading singular triplet (σ, u, v) via bilateral power iteration (fp32)."""
    if w.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions.")

    w = w.to(torch.float32)
    # v: (..., n, 1)
    v = torch.ones_like(w[..., :1, :].transpose(-2, -1))
    for _ in range(steps):
        v = torch.nn.functional.normalize(
            w.transpose(-2, -1) @ (w @ v),
            dim=-2,
            eps=eps,
        )
    u = torch.nn.functional.normalize(w @ v, dim=-2, eps=eps)
    # s: leading singular value
    s = (u.transpose(-2, -1) @ w @ v).squeeze(-1).squeeze(-1)

    return s, u, v


# =========================
# 误差指标
# =========================

@torch.no_grad()
def rms_error(A: torch.Tensor, B: torch.Tensor) -> float:
    """
    元素级 RMS 误差:
        sqrt( mean( (A_ij - B_ij)^2 ) )
    """
    diff = (A.to(torch.float32) - B.to(torch.float32)).reshape(-1)
    return torch.sqrt(torch.mean(diff * diff)).item()


@torch.no_grad()
def benchmark_power_iteration(
    shapes = ((128, 128), (256, 256), (512, 2048), (1024, 4096)),
    steps: int = 200,
    num_mats: int = 5,
    device: str | None = None,
):
    """
    对若干随机矩阵：
      - 比较 power_iteration 估计的最大奇异值 vs SVD 的最大奇异值（奇异值 MAE）
      - 比较 rank-1 重构矩阵的 RMS 误差
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}, steps={steps}, num_mats={num_mats}")
    torch.manual_seed(0)

    for (m, n) in shapes:
        sv_abs_err_list = []   # |σ_pi - σ_svd|
        sv_rel_err_list = []   # |σ_pi - σ_svd| / σ_svd
        rms_list = []          # RMS( uσv^T (pi) - uσv^T (svd) )

        for _ in range(num_mats):
            # 随机矩阵；你想测更极端谱的话，这里可以自己造 U S V^T
            W = torch.randn(m, n, device=device, dtype=torch.float32)

            # ---- SVD ground truth ----
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)  # U:(m,k), S:(k,), Vh:(k,n)
            sigma_svd = S[0]
            # 取首个奇异向量，整理成列向量
            u_svd = U[:, [0]]                             # (m,1)
            v_svd = Vh[[0], :].transpose(-2, -1)          # (n,1)

            # rank-1 重构
            sigma_svd_mat = sigma_svd.view(1, 1)          # (1,1)，方便广播
            W_svd_rank1 = u_svd @ (v_svd.transpose(-2, -1) * sigma_svd_mat)  # (m,n)

            # ---- Power iteration ----
            sigma_pi, u_pi, v_pi = power_iteration(W, steps=steps)
            # 确保 shape 对齐
            if sigma_pi.ndim == 0:
                sigma_pi_mat = sigma_pi.view(1, 1)
            else:
                sigma_pi_mat = sigma_pi[..., None, None]

            W_pi_rank1 = u_pi @ (v_pi.transpose(-2, -1) * sigma_pi_mat)

            # ---- 指标 ----
            abs_err = (sigma_pi.to(torch.float32) - sigma_svd).abs().item()
            rel_err = abs_err / (sigma_svd.abs().item() + 1e-20)
            sv_abs_err_list.append(abs_err)
            sv_rel_err_list.append(rel_err)

            rms = rms_error(W_pi_rank1, W_svd_rank1)
            rms_list.append(rms)

        sv_abs_err_t = torch.tensor(sv_abs_err_list)
        sv_rel_err_t = torch.tensor(sv_rel_err_list)
        rms_t = torch.tensor(rms_list)

        print(f"\nshape {m}x{n}:")
        print(f"  σ abs error:  mean = {sv_abs_err_t.mean().item():.3e}, "
              f"max = {sv_abs_err_t.max().item():.3e}")
        print(f"  σ rel error:  mean = {sv_rel_err_t.mean().item():.3e}, "
              f"max = {sv_rel_err_t.max().item():.3e}")
        print(f"  rank-1 RMS:   mean = {rms_t.mean().item():.3e}, "
              f"max = {rms_t.max().item():.3e}")


if __name__ == "__main__":
    benchmark_power_iteration()
