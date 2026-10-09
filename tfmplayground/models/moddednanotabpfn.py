# Vendored from https://github.com/borawhocodess/modded-nanotabpfn (train_nano.py) at 6671842

import torch
import torch.nn.functional as F
from torch import nn

from tfmplayground.configs.models import ModdedNanoTabPFNClassifierConfig, ModdedNanoTabPFNRegressorConfig
from tfmplayground.models.base import TabularFoundationModel


class LowerPrecisionRMSNorm(nn.RMSNorm):
    """
    code adapted from: https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/architectures/tabpfn_v2_6.py
    """
    def forward(self, x):
        if x.dtype in (torch.float16, torch.bfloat16):
            with torch.amp.autocast("cuda", enabled=False):
                return super().forward(x)
        return super().forward(x)


class ThinkingRows(nn.Module):
    """
    code adapted from: https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/architectures/tabpfn_v2_6.py
    """
    def __init__(self, num_thinking_rows: int, e: int):
        super().__init__()
        self.num_thinking_rows = num_thinking_rows
        self.row_tokens = nn.Parameter(torch.empty(num_thinking_rows, e))
        nn.init.normal_(self.row_tokens)

    def forward(self, x, sep):
        b, r, c, e = x.shape
        thinking = self.row_tokens.unsqueeze(0).unsqueeze(2).expand(b, -1, c, -1)
        x = torch.cat([thinking, x], dim=1)
        sep = sep + self.num_thinking_rows
        return x, sep


# -----------------------------------------------------------------------------
# model


class ModdedNanoTabPFN(nn.Module):
    def __init__(self, l, a, e, h, o, residual_decay=1.0, thinking_rows=16, feature_group_size=3):
        """
        l : num layers
        a : num attention heads
        e : embedding size
        h : mlp hidden size
        o : num outputs
        residual_decay : exponential decay of residual stream per layer (1.0 = no decay)
        """
        super().__init__()
        self.l = l
        self.a = a
        self.e = e
        self.h = h
        self.o = o
        self.feature_encoder = FeatureEncoder(e, feature_group_size=feature_group_size)
        self.target_encoder = TargetEncoder(e)
        self.transformer_encoder = TransformerEncoderStack(l, a, e, h, residual_decay=residual_decay)
        self.decoder = Decoder(e, h, o)
        self.thinking_rows = ThinkingRows(num_thinking_rows=thinking_rows, e=e)

        self.register_buffer("borders", None, persistent=True)

    def forward(self, *args, **kwargs):
        if len(args) == 3:
            x = args[0]
            if args[2] is not None:
                x = torch.cat((x, args[2]), dim=1)
            return self._forward((x, args[1]), sep=args[0].shape[1], **kwargs)
        elif len(args) == 1 and isinstance(args[0], tuple):
            return self._forward(*args, **kwargs)

    def _forward(self, src, sep):
        x_src, y_src = src
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
        x_src = self.feature_encoder(x_src, sep)
        num_rows = x_src.shape[1]
        y_src = self.target_encoder(y_src, num_rows)
        src = torch.cat([x_src, y_src], 2)
        src, sep = self.thinking_rows(src, sep)
        output = self.transformer_encoder(src, sep)
        output = output[:, sep:, :-1, :].mean(dim=2)
        output = self.decoder(output)
        return output


class FeatureEncoder(nn.Module):
    def __init__(self, e, feature_group_size):
        super().__init__()
        self.feature_group_size = feature_group_size
        self.linear_layer = nn.Linear(feature_group_size, e)

    def forward(self, x, sep):
        n_cols = x.shape[-1]
        idxs = torch.arange(n_cols, dtype=torch.long, device=x.device)
        x = torch.stack([x[:, :, (idxs + (2 ** i - 1)) % n_cols] for i in range(self.feature_group_size)], dim=-1)
        mean = x[:, :sep].mean(dim=1, keepdim=True)
        std = x[:, :sep].std(dim=1, keepdim=True) + 1e-8
        x = (x - mean) / std
        x = torch.clip(x, min=-100, max=100)
        return self.linear_layer(x)


class TargetEncoder(nn.Module):
    def __init__(self, e):
        super().__init__()
        self.linear_layer = nn.Linear(1, e)

    def forward(self, y_train, num_rows):
        mean = y_train.mean(dim=1, keepdim=True)
        padding = mean.repeat(1, num_rows - y_train.shape[1], 1)
        y = torch.cat([y_train, padding], dim=1)
        y = y.unsqueeze(-1)
        return self.linear_layer(y)


class TransformerEncoderStack(nn.Module):
    def __init__(self, l, a, e, h, residual_decay=1.0):
        super().__init__()
        self.residual_decay = residual_decay
        self.transformer_blocks = nn.ModuleList()
        for _ in range(l):
            self.transformer_blocks.append(TransformerEncoderLayer(a, e, h))

    def forward(self, x, sep):
        for i, block in enumerate(self.transformer_blocks):
            x = x * (self.residual_decay ** i)
            x = block(x, sep=sep)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, a, e, h, eps=1e-5):
        super().__init__()
        self.num_heads = a
        self.head_dim = e // a
        assert e % a == 0, "Embedding size must be divisible by heads"

        self.qkv_datapoints = nn.Linear(e, 3 * e)
        self.qkv_features = nn.Linear(e, 3 * e)

        self.linear1 = nn.Linear(e, h)
        self.linear2 = nn.Linear(h, e)

        self.norm1 = LowerPrecisionRMSNorm(e, eps=eps)
        self.norm2 = LowerPrecisionRMSNorm(e, eps=eps)
        self.norm3 = LowerPrecisionRMSNorm(e, eps=eps)

    def forward(self, src, sep):
        b, r, c, e = src.shape

        x = src.reshape(b * r, c, e)
        res = x
        x = self.norm1(x)

        qkv = self.qkv_features(x)
        qkv = qkv.reshape(b * r, c, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(b * r, c, e)

        src = res + x
        src = src.reshape(b, r, c, e)

        x = src.transpose(1, 2).reshape(b * c, r, e)
        res = x
        x = self.norm2(x)

        qkv = self.qkv_datapoints(x)
        qkv = qkv.reshape(b * c, r, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q_left, q_right = q.split([sep, r - sep], dim=2)

        k_train = k[:, :, :sep, :]
        v_train = v[:, :, :sep, :]

        x_left = F.scaled_dot_product_attention(q_left, k_train, v_train)
        x_right = F.scaled_dot_product_attention(q_right, k_train, v_train)

        x = torch.cat([x_left, x_right], dim=2)
        x = x.transpose(1, 2).reshape(b * c, r, e)

        src = res + x
        src = src.reshape(b, c, r, e).transpose(2, 1)

        x = self.norm3(src)
        x = self.linear2(F.gelu(self.linear1(x)))
        src = src + x

        return src


class Decoder(nn.Module):
    def __init__(self, e, h, o):
        super().__init__()
        self.linear1 = nn.Linear(e, h)
        self.linear2 = nn.Linear(h, o)

    def forward(self, x):
        return self.linear2(F.gelu(self.linear1(x)))


class ModdedNanoTabPFNModel(ModdedNanoTabPFN, TabularFoundationModel):
    """
    adapts moddednanotabpfn to tabularfoundationmodel interface
    """

    def __init__(self, config: ModdedNanoTabPFNClassifierConfig | ModdedNanoTabPFNRegressorConfig) -> None:
        """
        builds moddednanotabpfn from config and reserves its borders
        """
        self.config = config
        super().__init__(
            l=config.l,
            a=config.a,
            e=config.e,
            h=config.h,
            o=config.o,
            residual_decay=config.residual_decay,
            thinking_rows=config.thinking_rows,
            feature_group_size=config.feature_group_size,
        )
        self.register_buffer("borders", torch.zeros(config.o + 1))

    def forward(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> torch.Tensor:
        """
        takes train rows as context and predicts test rows through vendored forward
        """
        return super().forward(X_train, y_train, X_test)
