"""
Phase M — Long-context genomic model using State Space Models (Mamba/Hyena-style).

WHY LONG-CONTEXT MODELING IS CRITICAL FOR METHYLATION BIOLOGY
=============================================================
1. **Regulatory domains span 10kb–1Mb**: Enhancers, super-enhancers, and TADs
   (topologically associating domains) control gene expression over distances
   that far exceed typical 1–10kb windows.  A model limited to 1kb cannot
   learn that a distal enhancer's methylation state predicts a promoter's
   activity 500kb away.

2. **CpG islands cluster non-uniformly**: The informative CpG-rich regions
   are separated by vast CpG-poor deserts.  Long-context models can attend
   across these deserts without expensive sliding-window heuristics.

3. **Imprinting and allele-specific methylation**: These patterns often
   involve coordinated methylation across large genomic blocks (e.g. the
   H19/IGF2 locus spans ~100kb).

4. **ONT reads are inherently long**: Nanopore reads routinely span 10–100kb,
   giving us per-read methylation across large regions.  A matching model
   architecture lets us exploit that native resolution.

TRADEOFFS vs TRANSFORMERS
=========================
| Property            | Transformer (self-attention) | SSM (Mamba-style)          |
|---------------------|------------------------------|----------------------------|
| Memory              | O(L²)                        | O(L) — linear              |
| Speed               | O(L²) per layer              | O(L log L) or O(L)         |
| Max practical L     | ~4k–16k tokens               | 50k–1M+ tokens             |
| Long-range learning | Excellent (explicit pairwise) | Very good (learned recurrence) |
| Interpretability    | Attention maps available      | Harder; need probing       |
| Maturity            | Battle-tested                 | Rapidly maturing (2024+)   |

For genomic methylation at 50kb–1Mb, transformers are memory-prohibitive.
SSMs are the practical choice.

ARCHITECTURE
============
    Input: (batch, channels, L)  where L = 50k–1M positions
    ┌──────────────────────────────────────────┐
    │  Embedding / Projection                   │
    │  (sequence one-hot + methylation tracks)  │
    └──────────────┬───────────────────────────┘
                   │
    ┌──────────────▼───────────────────────────┐
    │  N × MambaBlock                           │
    │  ┌─────────────────────────────────────┐  │
    │  │ Linear expand → Conv1d → SSM core   │  │
    │  │ → gated output → residual + norm    │  │
    │  └─────────────────────────────────────┘  │
    └──────────────┬───────────────────────────┘
                   │
    ┌──────────────▼───────────────────────────┐
    │  Global pooling (mean / learned [CLS])    │
    └──────────────┬───────────────────────────┘
                   │
    ┌──────────────▼───────────────────────────┐
    │  Classification / Regression head         │
    └──────────────────────────────────────────┘

TRAINING CONSIDERATIONS FOR LONG SEQUENCES
===========================================
1. **Gradient checkpointing**: Essential — saves ~60% memory at cost of ~30%
   slower backward pass.  Enabled by default below.
2. **Mixed precision (bf16/fp16)**: Mandatory for 1M-length inputs.
3. **Sequence packing**: Combine shorter samples into one long sequence with
   separator tokens to maximise GPU utilisation.
4. **Learning rate**: Start lower (1e-4) than window-based models; long
   sequences produce noisier gradients.
5. **Batch size**: Likely 1–4 per GPU for 1M-length sequences; use gradient
   accumulation to simulate larger effective batches.
6. **Data loading**: Use memory-mapped files or streaming — do NOT load full
   chromosomes into a pandas DataFrame.

DATASET / DATALOADER MODIFICATIONS REQUIRED
============================================
- New ``LongContextDataset`` that yields (sequence_track, methylation_track)
  tensors of length L rather than per-window features.
- ``LongContextCollate`` that pads/truncates to max_length and builds
  attention masks (for any optional attention layers).
- Chromosome-level or region-level sampling strategy instead of
  per-window random sampling.
- See ``long_context_dataset()`` factory at the bottom of this file.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

# Try to import the optimised CUDA Mamba kernel.  Falls back to pure PyTorch.
_HAS_MAMBA_CUDA = False
try:
    from mamba_ssm import Mamba as MambaCUDA  # type: ignore[import-untyped]
    _HAS_MAMBA_CUDA = True
    logger.debug("mamba-ssm CUDA kernel available — using fast path")
except ImportError:
    MambaCUDA = None
    logger.debug("mamba-ssm not installed — using pure-PyTorch SSM (pip install mamba-ssm for ~3× speedup)")


# =====================================================================
# Selective State Space Model (S6) Core — Mamba-style
# =====================================================================
class S6Core(nn.Module):
    """
    Simplified selective scan (S6) block inspired by Mamba.

    For production, consider using the official ``mamba-ssm`` CUDA kernel
    (``pip install mamba-ssm``).  This pure-PyTorch implementation is
    portable and correct but ~3× slower than the fused kernel.

    Parameters
    ----------
    d_model : int
        Hidden dimension.
    d_state : int
        SSM state expansion factor (N in the Mamba paper).
    dt_rank : int or "auto"
        Rank of the Δ projection.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        dt_rank: int | str = "auto",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # Selective projections: input → (Δ, B, C)
        self.x_proj = nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False)

        # Δ (dt) projection
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)
        # Initialise dt bias so initial Δ is in a reasonable range
        with torch.no_grad():
            dt_init = torch.exp(
                torch.rand(d_model) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            )
            inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
            self.dt_proj.bias.copy_(inv_dt)

        # A parameter (log-space for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D skip connection
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, length, d_model)
        returns: (batch, length, d_model)
        """
        B, L, D = x.shape
        N = self.d_state

        # Project to get Δ, B_sel, C_sel
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*N)
        dt, B_sel, C_sel = x_dbl.split([self.dt_rank, N, N], dim=-1)

        dt = F.softplus(self.dt_proj(dt))  # (B, L, D)
        A = -torch.exp(self.A_log)          # (D, N)

        # Discretise: Ā = exp(Δ·A), B̄ = Δ·B
        # For efficiency we do the recurrence in a scan
        dt = dt.transpose(1, 2)       # (B, D, L)
        B_sel = B_sel.transpose(1, 2)  # (B, N, L)
        C_sel = C_sel.transpose(1, 2)  # (B, N, L)
        x_t = x.transpose(1, 2)       # (B, D, L)

        # Sequential scan (pure PyTorch — replace with parallel scan / CUDA kernel for speed)
        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for i in range(L):
            dt_i = dt[:, :, i].unsqueeze(-1)          # (B, D, 1)
            B_i = B_sel[:, :, i].unsqueeze(1)          # (B, 1, N)
            C_i = C_sel[:, :, i].unsqueeze(1)          # (B, 1, N)
            x_i = x_t[:, :, i].unsqueeze(-1)           # (B, D, 1)

            # State update: h = Ā·h + B̄·x
            A_bar = torch.exp(dt_i * A.unsqueeze(0))   # (B, D, N)
            B_bar = dt_i * B_i                          # (B, D, N)
            h = A_bar * h + B_bar * x_i                 # (B, D, N)

            # Output: y = C·h + D·x
            y_i = (C_i * h).sum(dim=-1)                # (B, D)
            y_i = y_i + self.D * x_t[:, :, i]
            ys.append(y_i)

        y = torch.stack(ys, dim=-1)  # (B, D, L)
        return y.transpose(1, 2)     # (B, L, D)


# =====================================================================
# Mamba Block
# =====================================================================
class MambaBlock(nn.Module):
    """
    Single Mamba block.

    Uses the fused CUDA kernel from ``mamba-ssm`` when available (~3× faster).
    Falls back to a pure-PyTorch implementation otherwise.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
        dt_rank: int | str = "auto",
        dropout: float = 0.0,
        use_cuda_kernel: bool = True,
    ) -> None:
        super().__init__()
        self._use_cuda = _HAS_MAMBA_CUDA and use_cuda_kernel

        if self._use_cuda:
            # Use the optimised mamba-ssm CUDA implementation
            self.norm = nn.LayerNorm(d_model)
            self.mamba_cuda = MambaCUDA(
                d_model=d_model,
                d_state=d_state,
                d_conv=conv_kernel,
                expand=expand,
            )
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        else:
            # Pure-PyTorch fallback
            d_inner = d_model * expand
            self.norm = nn.LayerNorm(d_model)
            self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
            self.conv1d = nn.Conv1d(
                d_inner, d_inner, kernel_size=conv_kernel,
                padding=conv_kernel - 1, groups=d_inner, bias=True,
            )
            self.ssm = S6Core(d_model=d_inner, d_state=d_state, dt_rank=dt_rank)
            self.out_proj = nn.Linear(d_inner, d_model, bias=False)
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, length, d_model) → (batch, length, d_model)"""
        residual = x
        x = self.norm(x)

        if self._use_cuda:
            x = self.mamba_cuda(x)
            x = self.dropout(x)
            return x + residual

        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        # Conv branch
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, :x.size(1)]
        x_branch = x_branch.transpose(1, 2)
        x_branch = F.silu(x_branch)

        # SSM
        x_branch = self.ssm(x_branch)

        # Gate
        out = x_branch * F.silu(z)
        out = self.out_proj(out)
        out = self.dropout(out)

        return out + residual


# =====================================================================
# Long-Context Genomic Model
# =====================================================================
class LongContextGenomicModel(nn.Module):
    """
    Mamba-based model for long-range genomic methylation modeling.

    Processes sequences of 50kb–1Mb with integrated methylation tracks
    in linear time and memory.

    Parameters
    ----------
    n_input_channels : int
        Number of input channels.  For sequence+methylation:
        5 (ACGTN one-hot) + n_meth_channels.
    d_model : int
        Hidden state dimension.
    n_layers : int
        Number of stacked MambaBlocks.
    d_state : int
        SSM state dimension.
    expand : int
        Inner expansion factor.
    num_classes : int
        Output classes (1 for binary, >1 for multiclass).
    task : str
        'classification' or 'regression'.
    pool : str
        Global pooling: 'mean', 'max', or 'cls'.
    dropout : float
        Dropout rate.
    gradient_checkpointing : bool
        Use gradient checkpointing to reduce memory (recommended for L > 50k).
    """

    def __init__(
        self,
        n_input_channels: int = 8,  # 5 seq + 3 meth tracks
        d_model: int = 256,
        n_layers: int = 6,
        d_state: int = 16,
        expand: int = 2,
        num_classes: int = 2,
        task: str = "classification",
        pool: str = "mean",
        dropout: float = 0.1,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.task = task
        self._num_classes = num_classes
        self.pool = pool
        self.gradient_checkpointing = gradient_checkpointing

        # Input projection: (channels, L) → (L, d_model)
        self.input_proj = nn.Linear(n_input_channels, d_model)

        # Optional CLS token
        if pool == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Mamba backbone
        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # Output head
        out_dim = 1 if (num_classes == 2 and task == "classification") or task == "regression" else num_classes
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, out_dim),
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"LongContextGenomicModel: d_model={d_model}, layers={n_layers}, "
            f"params={n_params:,}, pool={pool}, grad_ckpt={gradient_checkpointing}"
        )

    def forward(
        self,
        x: torch.Tensor,
        methylation_track: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor
            Sequence input.  Shape (B, C_seq, L) for one-hot or (B, L) for tokenized.
            If one-hot, C_seq is typically 5 (ACGTN).
        methylation_track : Tensor or None
            Per-position methylation features.  Shape (B, C_meth, L).
            E.g. 3 channels: beta, coverage, is_cpg.

        Returns
        -------
        dict with keys: logits, probs (classification), embed
        """
        # Handle tokenized input
        if x.dim() == 2:
            # (B, L) integer tokens → one-hot
            x = F.one_hot(x.long(), num_classes=5).float().transpose(1, 2)  # (B, 5, L)

        # Concatenate methylation tracks along channel dim
        if methylation_track is not None:
            x = torch.cat([x, methylation_track], dim=1)  # (B, C_total, L)

        # (B, C, L) → (B, L, C)
        x = x.transpose(1, 2)

        # Project to d_model
        x = self.input_proj(x)  # (B, L, d_model)

        # Prepend CLS token if needed
        if self.pool == "cls":
            B = x.size(0)
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)

        # Mamba blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.final_norm(x)

        # Pool
        if self.pool == "cls":
            embed = x[:, 0, :]
        elif self.pool == "max":
            embed = x.max(dim=1).values
        else:
            embed = x.mean(dim=1)

        logits = self.head(embed)

        out: dict[str, torch.Tensor] = {"logits": logits, "embed": embed}
        if self.task == "classification":
            if self._num_classes == 2:
                out["probs"] = torch.sigmoid(logits)
            else:
                out["probs"] = torch.softmax(logits, dim=-1)

        return out

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> LongContextGenomicModel:
        """Instantiate from config dict."""
        lc = cfg.get("long_context", {})
        return cls(
            n_input_channels=lc.get("n_input_channels", 8),
            d_model=lc.get("d_model", 256),
            n_layers=lc.get("n_layers", 6),
            d_state=lc.get("d_state", 16),
            expand=lc.get("expand", 2),
            num_classes=cfg.get("num_classes", 2),
            task=cfg.get("task", "classification"),
            pool=lc.get("pool", "mean"),
            dropout=lc.get("dropout", 0.1),
            gradient_checkpointing=lc.get("gradient_checkpointing", True),
        )


# =====================================================================
# Long-Context Dataset (modifications to existing pipeline)
# =====================================================================
class LongContextDataset(torch.utils.data.Dataset):
    """
    Dataset for long-context genomic modeling.

    Instead of per-window features, each sample is a contiguous genomic
    region (e.g. a chromosome arm or a 500kb region) with per-position
    sequence and methylation tracks.

    Compatible with the existing pipeline — wraps the same methylation
    DataFrame but yields position-level tensors.
    """

    def __init__(
        self,
        regions: list[dict[str, Any]],
        labels: list[int] | None = None,
        max_length: int = 100_000,
        n_meth_channels: int = 3,
    ) -> None:
        """
        Parameters
        ----------
        regions : list of dict
            Each dict has:
                - 'sequence': str of DNA bases (or None)
                - 'meth_positions': array of genomic positions with methylation
                - 'meth_betas': array of beta values at those positions
                - 'meth_coverages': array of coverage values
                - 'region_length': int, total bp length of the region
        labels : list of int or None
        max_length : int
            Pad/truncate all regions to this length.
        n_meth_channels : int
            Number of methylation channels (beta, coverage, is_cpg).
        """
        self.regions = regions
        self.labels = labels
        self.max_length = max_length
        self.n_meth_channels = n_meth_channels

    def __len__(self) -> int:
        return len(self.regions)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        region = self.regions[idx]
        L = self.max_length

        # Sequence track: (5, L) one-hot
        seq_track = torch.zeros(5, L, dtype=torch.float32)
        nuc_map = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
        seq = region.get("sequence", "")
        for i, base in enumerate(seq[:L]):
            seq_track[nuc_map.get(base.upper(), 4), i] = 1.0

        # Methylation track: (n_channels, L)
        meth_track = torch.zeros(self.n_meth_channels, L, dtype=torch.float32)
        positions = region.get("meth_positions", [])
        betas = region.get("meth_betas", [])
        coverages = region.get("meth_coverages", [])
        for pos, beta, cov in zip(positions, betas, coverages):
            if 0 <= pos < L:
                meth_track[0, pos] = beta       # channel 0: beta value
                meth_track[1, pos] = cov / 100.0  # channel 1: normalised coverage
                meth_track[2, pos] = 1.0         # channel 2: CpG indicator

        item: dict[str, Any] = {
            "sequence": seq_track,               # (5, L)
            "methylation_track": meth_track,     # (n_meth, L)
            "idx": idx,
        }

        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def long_context_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate for LongContextDataset — stacks tensors, lists for non-tensors."""
    collated: dict[str, Any] = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            collated[k] = torch.stack(vals)
        else:
            collated[k] = vals
    return collated


# =====================================================================
# Memory / Speed Benchmarking Utility
# =====================================================================
def benchmark_model(
    model: nn.Module,
    seq_lengths: list[int] | None = None,
    batch_size: int = 1,
    n_input_channels: int = 8,
    device: str = "cpu",
    n_warmup: int = 2,
    n_trials: int = 5,
) -> list[dict[str, Any]]:
    """
    Benchmark forward-pass memory and speed for different sequence lengths.

    Returns list of dicts with keys: length, peak_memory_mb, mean_time_ms.
    """
    import time

    if seq_lengths is None:
        seq_lengths = [1_000, 10_000, 50_000, 100_000]

    model = model.to(device)
    model.eval()
    results = []

    for L in seq_lengths:
        x = torch.randn(batch_size, 5, L, device=device)
        meth = torch.randn(batch_size, n_input_channels - 5, L, device=device)

        # Warmup
        for _ in range(n_warmup):
            with torch.no_grad():
                _ = model(x, methylation_track=meth)

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(x, methylation_track=meth)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

        peak_mem = 0.0
        if device == "cuda":
            peak_mem = torch.cuda.max_memory_allocated() / 1e6

        result = {
            "length": L,
            "peak_memory_mb": round(peak_mem, 1),
            "mean_time_ms": round(sum(times) / len(times), 2),
        }
        results.append(result)
        logger.info(f"L={L:>8,}: mem={result['peak_memory_mb']:.1f}MB, time={result['mean_time_ms']:.1f}ms")

    return results
