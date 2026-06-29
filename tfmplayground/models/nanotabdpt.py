import math
from functools import wraps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear
from torch.nn.attention import SDPBackend, sdpa_kernel

from tfmplayground.models.base import TabularFoundationModel


def flash_context(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if getattr(self, "use_flash", False):
            assert torch.cuda.is_available(), "FlashAttention requires CUDA support"
            bf_support = torch.cuda.get_device_capability()[0] >= 8
            dtype = torch.bfloat16 if bf_support else torch.float16
            with (
                torch.autocast(device_type="cuda", dtype=dtype),
                sdpa_kernel(SDPBackend.FLASH_ATTENTION),
            ):
                return func(self, *args, **kwargs)
        else:
            return func(self, *args, **kwargs)

    return wrapper


def maskmean(x, mask, dim):
    x = torch.where(mask, x, 0)
    return x.sum(dim=dim, keepdim=True) / mask.sum(dim=dim, keepdim=True)


def maskstd(x, mask, dim=0):
    num = mask.sum(dim=dim, keepdim=True)
    mean = maskmean(x, mask, dim=0)
    diffs = torch.where(mask, mean - x, 0)
    return ((diffs**2).sum(dim=0, keepdim=True) / (num - 1)) ** 0.5


def normalize_data(data, eval_pos=-1, dim=0, return_mean_std: bool = False):
    X = data[:eval_pos] if eval_pos > 0 else data
    mask = ~torch.isnan(X)
    mean = maskmean(X, mask, dim=dim)
    std = maskstd(X, mask, dim=dim) + 1e-6
    data = (data - mean) / std
    if return_mean_std:
        return data, mean, std
    return data


def clip_outliers(data, eval_pos=-1, n_sigma=4, dim=0):
    X = data[:eval_pos] if eval_pos > 0 else data
    mask = ~torch.isnan(X)
    mean = maskmean(X, mask, dim=dim)
    cutoff = n_sigma * maskstd(X, mask, dim=dim)
    mask &= cutoff >= torch.abs(X - mean)
    cutoff = n_sigma * maskstd(X, mask, dim=dim)
    return torch.clip(data, mean - cutoff, mean + cutoff)


class TabDPTModel(TabularFoundationModel):
    def __init__(
        self,
        dropout: float,
        enc_cell_dim: int,
        n_out: int,
        regression_bin_count: int,
        regression_bin_min: float,
        regression_bin_max: float,
        nhead: int,
        nhid: int,
        ninp: int,
        nlayers: int,
        num_features: int,
        base_len: int,
        max_len: int,
        y_encoder_dim: int,
        num_col_attn_layers: int = 2,
        n_thinking_rows: int = 0,
        classification: bool = True,
        use_flash: bool = False,
        clip_sigma: float = 8.
    ):
        super().__init__()
        assert regression_bin_count >= 3, "regression_bin_count must be at least 3"
        assert regression_bin_min < regression_bin_max, "regression_bin_min must be smaller than regression_bin_max"
        self.n_out = n_out  # number of classification outputs
        self.regression_bin_count = regression_bin_count
        self.regression_bin_min = regression_bin_min
        self.regression_bin_max = regression_bin_max
        self.ninp = ninp  # embedding dimension
        self.nlayers = nlayers  # store for stochastic depth calculation
        self.n_thinking_rows = n_thinking_rows
        self.transformer_encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=ninp,
                    num_heads=nhead,
                    ff_dim=nhid,
                    base_len=base_len,
                    max_len=max_len,
                    y_encoder_dim=y_encoder_dim,
                )
                for _ in range(nlayers)
            ]
        )
        self.num_features = num_features
        self.encoder = nn.Linear(num_features, ninp)
        self.enc_norm = nn.LayerNorm(ninp, elementwise_affine=False)
        self.dropout = nn.Dropout(p=dropout)
        self.y_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, 2 * y_encoder_dim),
                    nn.GELU(),
                    nn.Linear(2 * y_encoder_dim, y_encoder_dim),
                    nn.LayerNorm(y_encoder_dim, elementwise_affine=False),
                )
                for _ in range(nlayers)
            ]
        )
        self.head = nn.Sequential(nn.Linear(ninp, nhid), nn.GELU(), nn.Linear(nhid, n_out + regression_bin_count))
        if n_thinking_rows > 0:
            self.thinking_embed = nn.Parameter(torch.empty(n_thinking_rows, ninp))
            nn.init.normal_(self.thinking_embed, std=0.02)
        self.use_flash = use_flash
        self.clip_sigma = clip_sigma
        self.classification = classification

    @flash_context
    def forward(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor) -> torch.Tensor:
        x_src = torch.cat([X_train, X_test], dim=1)
        if x_src.shape[-1] < self.num_features:
            x_src = F.pad(x_src, (0, self.num_features - x_src.shape[-1]))
        y_src = y_train

        x_src = x_src.transpose(0, 1)
        y_src = y_src.transpose(0, 1)
        eval_pos = y_src.shape[0]
        n_think = self.n_thinking_rows

        # preproces features by normalizing and clipping outliers
        x_src = clip_outliers(x_src, -1 if self.training else eval_pos, n_sigma=self.clip_sigma)
        x_src = normalize_data(x_src, -1 if self.training else eval_pos)
        x_src = clip_outliers(x_src, -1 if self.training else eval_pos, n_sigma=self.clip_sigma)
        # Replace NaN and Inf with 0 (inf can occur from division by zero in normalize_data)
        x_src = torch.nan_to_num(x_src, nan=0.0, posinf=0.0, neginf=0.0)

        x_src = self.encoder(x_src)
        src = self.enc_norm(x_src)
        if n_think > 0:
            B = src.shape[1]
            src = torch.cat([self.thinking_embed.unsqueeze(1).expand(n_think, B, -1), src], dim=0)

        for i, layer in enumerate(self.transformer_encoder):
            y_emb = self.y_encoders[i](y_src.unsqueeze(-1))
            if n_think > 0:
                B = y_emb.shape[1]
                y_emb = torch.cat([y_emb.new_zeros(n_think, B, y_emb.shape[-1]), y_emb], dim=0)
            # Layer returns residual only
            residual = layer(src, y_emb, eval_pos + n_think)
            src = src + residual

        # final head
        pred = self.head(src[eval_pos + n_think :].float())
        pred = pred.transpose(0, 1)
        if self.classification:
            return pred[..., : self.n_out]
        return pred[..., self.n_out :]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use float32 for computation to avoid bf16 underflow/NaNs
        variance = x.float().pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps)).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, bias: bool = False):
        super().__init__()
        self.up = Linear(d_model, 2 * ff_dim, bias=bias)
        self.down = Linear(ff_dim, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(u) * v)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        base_len: int,
        max_len: int,
        y_encoder_dim: int,
    ) -> None:
        """
        Args:
            embed_dim (int): Dimension of the embedding.
            num_heads (int): Number of attention heads.
            ff_dim (int): Dimension of the feed-forward network.
            base_len (int): Base length for attention scaling.
            max_len (int): Maximum length for attention scaling. If equal to base_len, attention scaling is disabled.
            y_encoder_dim (int): Dimension of per-layer y embedding; v_proj input is embed_dim + y_encoder_dim.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_gate = nn.Linear(embed_dim, num_heads, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim + y_encoder_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_norm = LayerNorm(embed_dim)
        self.ff_norm = LayerNorm(embed_dim)
        self.ff = SwiGLU(embed_dim, ff_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        # Attention temperature stuff
        # If base_len == max_len, disable attention scaling (kappa = 0 ensures scale is always 1.0)
        self.disable_attention_scaling = base_len == max_len
        dk = float(self.head_dim)
        self.register_buffer("n0", torch.tensor(float(base_len)))
        self.register_buffer("max_len_f", torch.tensor(float(max_len)))
        if self.disable_attention_scaling:
            # Set kappa to 0 to ensure scaling is always 1.0
            self.register_buffer("kappa", torch.tensor(0.0))
        else:
            self.register_buffer(
                "kappa",
                torch.tensor((math.sqrt(dk) - 1.0) / math.log(max_len / base_len)),
            )

        # Zero-init output projections so residual branch is identity at start (attn=0, ff=0)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.q_gate.weight)
        nn.init.zeros_(self.ff.down.weight)

    def get_scale_param(self, eval_pos: int, *, device, dtype) -> torch.Tensor:
        if self.disable_attention_scaling:
            return torch.tensor(1.0, device=device, dtype=dtype)
        n = torch.as_tensor(eval_pos, device=device, dtype=dtype)
        n = torch.minimum(n, self.max_len_f.to(dtype))
        beta = 1.0 + self.kappa.to(dtype) * torch.clamp(torch.log(n / self.n0.to(dtype)), min=0.0)
        return beta

    def forward(self, x: torch.Tensor, y: torch.Tensor, eval_pos: int) -> torch.Tensor:
        """Compute the residual for this layer.

        Args:
            x (torch.tensor): Input tensor of shape (L, B, D).
            y (torch.tensor): Target embedding for the context region of shape (eval_pos, B, y_encoder_dim).
            eval_pos (int): Number of context positions (length of y and K/V).
        Returns:
            torch.tensor: Residual tensor to be added to input (same shape as input).
        """
        # switch to (B, L, D) for attention computation
        x = x.transpose(0, 1)
        y = y.transpose(0, 1)
        B, L, _ = x.size()

        h = self.attn_norm(x)
        q = self.q_proj(h)
        gate = torch.sigmoid(self.q_gate(q))
        k = self.k_proj(h[:, :eval_pos])

        h_ctx = h[:, :eval_pos]
        v_in = torch.cat([h_ctx, y], dim=-1)
        v = self.v_proj(v_in)

        # reshape: (B, L, D) -> (B, num_heads, _, head_dim)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, eval_pos, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, eval_pos, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)
        beta = self.get_scale_param(eval_pos, device=q.device, dtype=q.dtype)
        default_scale = 1.0 / math.sqrt(self.head_dim)
        custom_scale = default_scale * beta

        attn = F.scaled_dot_product_attention(q, k, v, scale=custom_scale).transpose(1, 2)
        attn = attn * gate.unsqueeze(-1)
        attn = self.out_proj(attn.reshape(B, L, self.num_heads * self.head_dim))
        residual = attn + self.ff(self.ff_norm(x + attn))

        # back to (L, B, D) for output
        return residual.transpose(0, 1)
