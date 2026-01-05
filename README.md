# Fully **μP** Aligned LLM Training on Spectral Sphere

<div align="center">
  <a href="https://github.com/Unakar/Megatron-LM/tree/spectral_ball"><img src="https://www.nvidia.com/favicon.ico" height="16" width="16" style="vertical-align:middle"> <b>Megatron-LM</b></a>  |  
  <a href="https://wandb.ai/rqn17762075640-ustc/optimizer_arena_v2"><img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-dots-logo.svg" height="16" width="16" style="vertical-align:middle"> <b>WandB</b></a>  |  
  <a href="https://huggingface.co/collections/unakar666/spectral-sphere-optimizer"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="16" width="16" style="vertical-align:middle"> <b>HuggingFace</b></a>
</div>

## Code Structure

1. `megatron_scripts` : Scripts used for experiments
2. `plot_figures` : Code for plotting figures
   1. `precision` : Check precision of lambda-solver
   2. `overhead` : Check Overhead of lambda-solver
   3. `loss` : Code for plotting loss figures
   
## Usage

```bash
uv sync
uv run -m plot_figures.loss.plot_moe_val_loss
```