import math
import warnings
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.transformer import LayerNorm, Linear, MultiheadAttention
from torch.utils.checkpoint import checkpoint

# Fixed seed of the per-column random vectors (see NanoTabPFNModel.add_column_embeddings).
COLUMN_EMBEDDING_SEED = 42


class NanoTabPFNModel(nn.Module):
    def __init__(
        self, embedding_size: int, num_attention_heads: int, mlp_hidden_size: int, num_layers: int, num_outputs: int
    ):
        """Initializes the feature/target encoder, transformer blocks and decoder"""
        super().__init__()
        self.embedding_size = embedding_size
        self.num_attention_heads = num_attention_heads
        self.mlp_hidden_size = mlp_hidden_size
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.feature_encoder = FeatureEncoder(embedding_size)
        # Projects a fixed random vector per feature column into the embedding space (TabPFNv2's
        # "subspace" feature positional embedding), see add_column_embeddings.
        self.column_embedding = nn.Linear(embedding_size // 4, embedding_size)
        self.target_encoder = TargetEncoder(embedding_size)
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(
                TransformerEncoderLayer(embedding_size, num_attention_heads, mlp_hidden_size)
            )
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)

        # Opt-in feature extraction: when save_embeddings is True, _forward stores the
        # per-row target-token embedding (before the decoder) in embeddings. Off by default,
        # so the normal prediction path is unchanged.
        self.save_embeddings = False
        self.embeddings: torch.Tensor | None = None
        
        # Opt-in activation checkpointing: during training, store only each block's input
        # and recompute the block in the backward pass. Trades ~30% compute for memory.
        self.gradient_checkpointing = False

    # TODO: consider getting rid of this and just provide a single interface
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Provides two interfaces:
        model(X_train, y_train, X_test)
            Args:
                X_train: (torch.Tensor) a tensor of shape (batch_size, num_train_datapoints, num_features)
                y_train: (torch.Tensor) a tensor of shape (batch_size, num_train_datapoints, 1)
                X_test: (torch.Tensor) a tensor of shape (batch_size, num_test_datapoints, num_features)

        model((x,y), train_test_split_index)
            Args:
                x: (torch.Tensor) a tensor of shape (batch_size, num_datapoints, num_features)
                y: (torch.Tensor) a tensor of shape (batch_size, num_train_datapoints, 1)


        The former is similar to the sklearn interface.
        In the latter x is the concatenation of X_train and X_test, y is y_train and
        train_test_split_index is the length of X_train.
        Our model internally works with the latter representation, so we convert the former into
        the latter and forward it to _forward.

        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_test_datapoints, num_classes),
                           which represent the predicted logits
        """
        if len(args) == 3:
            # case model(train_x, train_y, test_x)
            x = args[0]
            if args[2] is not None:
                x = torch.cat((x, args[2]), dim=1)
            return self._forward((x, args[1]), train_test_split_index=args[0].shape[1], **kwargs)
        elif len(args) == 1 and isinstance(args[0], tuple):
            # case model((x,y), train_test_split_index=None)
            return self._forward(*args, **kwargs)

    def add_column_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Adds a fixed random vector per feature column, projected by a learned linear layer, as
        TabPFNv2 does, so that columns with similar values stay distinguishable in the attention
        between features. The vectors come from a fixed seed and are generated on the CPU, so they
        are identical across calls and devices (a model trained on GPU sees the same vectors on
        CPU). Training shuffles the column order, so they act as random column identifiers.

        Args:
            x: (torch.Tensor) feature embeddings of shape (batch_size, num_rows, num_features, embedding_size)
        Returns:
            (torch.Tensor) same shape, with each column's vector added to all of its cells
        """
        generator = torch.Generator().manual_seed(COLUMN_EMBEDDING_SEED)
        embs = torch.randn(x.shape[2], self.embedding_size // 4, generator=generator).to(x.device)
        return x + self.column_embedding(embs).to(x.dtype)[None, None]

    def _forward(
        self, src: tuple[torch.Tensor, torch.Tensor], train_test_split_index: int, num_mem_chunks: int = 1
    ) -> torch.Tensor:
        x_src, y_src = src
        # we expect the labels to look like (batches, num_train_datapoints, 1),
        # so we add the last dimension if it is missing
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
        # from here on B=Batches, R=Rows, C=Columns, E=embedding size
        # converts scalar values to embeddings, so (B,R,C-1) -> (B,R,C-1,E)
        x_src = self.feature_encoder(x_src, train_test_split_index)
        x_src = self.add_column_embeddings(x_src)
        num_rows = x_src.shape[1]
        # padds the y_train up to y by using the mean,
        # then converts scalar values to embeddings (B,R,1,E)
        y_src = self.target_encoder(y_src, num_rows)
        # concatenates the feature embeddings with the target embeddings
        # to give us the full table of embeddings (B,R,C,E))
        src = torch.cat([x_src, y_src], 2)
        # repeatedly applies the transformer block on (B,R,C,E)
        for block in self.transformer_blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                # Checkpointing keeps each block's input alive until the backward pass. Under
                # autocast that input is fp32 (it is a LayerNorm output), so cast it to the
                # autocast dtype first: half the stored memory. Only affects checkpointed
                # mixed-precision training; inference and fp32 training are unchanged.
                device_type = src.device.type
                if torch.is_autocast_enabled(device_type):
                    src = src.to(torch.get_autocast_dtype(device_type))
                src = checkpoint(block, src, train_test_split_index, use_reentrant=False)
            else:
                src = block(src, train_test_split_index=train_test_split_index, num_mem_chunks=num_mem_chunks)
        if self.save_embeddings:
            # per-row target-token embedding (B, R, E), before the decoder. Covers both train
            # and test rows; the caller slices whichever it needs.
            self.embeddings = src[:, :, -1, :].detach()
        # selects the target embeddings (B,num_targets,1,E)
        output = src[:, train_test_split_index:, -1, :]
        # runs the embeddings through the decoder to get
        # the logits of our predictions (B,num_targets,num_classes)
        output = self.decoder(output)
        return output


def normalize_features(x: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
    """
    Normalizes each feature based on the mean and std of the training rows (ignoring
    missing entries), imputes the missing entries and clips outliers to [-100, 100].
    Emits a per-cell missing indicator alongside the value, so the embedding sees
    "value + was-missing" as one unit per feature. This is the feature preprocessing,
    kept separate from the embedding so it can be reused and tested on its own.

    Args:
        x: (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features)
        train_test_split_index: (int) the number of datapoints in X_train; the stats
                                use only x[:, :train_test_split_index]
    Returns:
        (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, 2), whose
                       last axis is [normalized_value, missing_indicator]
    """
    x = x.unsqueeze(-1)
    # Indicator is generated BEFORE imputing: 1.0 where the cell was missing, else 0.0.
    indicator = torch.isnan(x).to(x.dtype)
    train = x[:, :train_test_split_index]
    mean = torch.nanmean(train, dim=1, keepdims=True)
    # Unbiased std that ignores NaNs: sum of squared deviations over (count - 1).
    count = (~torch.isnan(train)).sum(dim=1, keepdims=True)
    sum_sq = torch.nansum((train - mean) ** 2, dim=1, keepdims=True)
    std = torch.sqrt(sum_sq / (count - 1)) + torch.finfo(torch.float32).eps
    # clip() cannot rescue a non-finite std (clamp of NaN is NaN), so a single
    # training row (count - 1 == 0, std is NaN) falls back to a std of 1.0.
    std = torch.where(torch.isfinite(std), std, torch.ones_like(std))
    x = (x - mean) / std
    # Impute the holes to 0, i.e. the train mean after normalization.
    x = torch.nan_to_num(x, nan=0.0)
    x = torch.clip(x, min=-100, max=100)
    return torch.cat([x, indicator], dim=-1)

class FeatureEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        """Creates the linear layer that we will use to embed our features.

        Input width is 2 because normalize_features emits [value, missing_indicator]
        per cell, so value and indicator are projected together into one embedding.
        """
        super().__init__()
        self.linear_layer = nn.Linear(2, embedding_size)

    def forward(self, x: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        """
        Normalizes the features (see normalize_features) and applies a linear layer to
        embed them.

        Args:
            x: (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features)
            train_test_split_index: (int) the number of datapoints in X_train
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size), representing
                           the embeddings of the features
        """
        return self.linear_layer(normalize_features(x, train_test_split_index))


def pad_targets(y_train: torch.Tensor, num_rows: int) -> torch.Tensor:
    """
    Pads y_train up to the full row count by filling the (unknown) test positions with
    the per-dataset train-label mean, and adds a per-row indicator that is 1.0 on the test
    positions, as TabPFNv2 does. Without it a test row whose placeholder equals a real label
    (e.g. mean 1.0 with classes 0, 1, 2) would look exactly like a train row of that class.
    This is the target preprocessing, kept separate from the embedding so it can be reused
    and tested on its own.

    Args:
        y_train: (torch.Tensor) a tensor of shape (batch_size, num_train_datapoints, 1)
        num_rows: (int) the full length of y (train + test)
    Returns:
        (torch.Tensor) a tensor of shape (batch_size, num_rows, 1, 2), whose last axis is
                       [padded_value, is_test_indicator]
    """
    mean = torch.mean(y_train, axis=1, keepdim=True)
    padding = mean.repeat(1, num_rows - y_train.shape[1], 1)
    y = torch.cat([y_train, padding], dim=1)  # (B, R, 1)
    is_test = torch.zeros_like(y)
    is_test[:, y_train.shape[1]:] = 1.0
    return torch.cat([y, is_test], dim=-1).unsqueeze(2)  # (B, R, 1, 2)


class TargetEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        """Creates the linear layer that we will use to embed our targets.

        Input width is 2 because pad_targets emits [value, is_test_indicator] per row.
        """
        super().__init__()
        self.linear_layer = nn.Linear(2, embedding_size)

    def forward(self, y_train: torch.Tensor, num_rows: int) -> torch.Tensor:
        """
        Pads y_train up to the full length (see pad_targets) and embeds it with a linear layer.

        Args:
            y_train: (torch.Tensor) a tensor of shape (batch_size, num_train_datapoints, 1)
            num_rows: (int) the full length of y
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows, 1, embedding_size), representing
                           the embeddings of the targets
        """
        return self.linear_layer(pad_targets(y_train, num_rows))


class TransformerEncoderLayer(nn.Module):
    """
    Modified version of older version of https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/transformer.py#L630
    """

    def __init__(
        self,
        embedding_size: int,
        nhead: int,
        mlp_hidden_size: int,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.self_attention_between_datapoints = MultiheadAttention(
            embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype
        )
        self.self_attention_between_features = MultiheadAttention(
            embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype
        )

        self.linear1 = Linear(embedding_size, mlp_hidden_size, device=device, dtype=dtype)
        self.linear2 = Linear(mlp_hidden_size, embedding_size, device=device, dtype=dtype)

        self.norm1 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm3 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)

        # Opt-in interpretability: when save_feature_attention is True, forward stores the
        # target column's attention to every feature (averaged over samples and heads) in
        # feature_attention. Off by default, so the normal forward path is unchanged.
        self.save_feature_attention = False
        self.feature_attention: torch.Tensor | None = None

    @staticmethod
    def target_attention(attention: MultiheadAttention, x: torch.Tensor) -> torch.Tensor:
        """Attention weights of the last token (the target column) over all tokens, averaged over
        heads, for every row: (N, C) for x of shape (N, C, E). Only that row of the attention matrix
        is computed, so memory grows with C instead of C^2 (TabPFN-Wide computes the full matrix one
        sample at a time). Equals need_weights=True (head-averaged) sliced at [:, -1]."""
        n, c, e = x.shape
        h = attention.num_heads
        d = e // h
        w_q, w_k, _ = attention.in_proj_weight.chunk(3)
        b_q, b_k, _ = attention.in_proj_bias.chunk(3)
        q = F.linear(x[:, -1:, :], w_q, b_q).view(n, 1, h, d).transpose(1, 2)  # (N, H, 1, d)
        k = F.linear(x, w_k, b_k).view(n, c, h, d).transpose(1, 2)  # (N, H, C, d)
        weights = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(d), dim=-1)  # (N, H, 1, C)
        return weights.mean(dim=1)[:, 0, :]

    def forward(self, src: torch.Tensor, train_test_split_index: int, num_mem_chunks: int = 1) -> torch.Tensor:
        """
        Takes the embeddings of the table as input and applies self-attention between features
        and self-attention between datapoints followed by a simple 2 layer MLP.

        Args:
            src: (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size)
                                that contains all the embeddings for all the cells in the table
            train_test_split_index: (int) the length of X_train
            num_mem_chunks: (int) Number of chunks that memory-intense operations will be split into.
                                  Higher values use less memory but are slower. Needs to be set to 1
                                  during training to get correct gradients.
        Returns
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size)
        """
        batch_size, rows_size, col_size, embedding_size = src.shape
        # attention between features
        src = src.reshape(batch_size * rows_size, col_size, embedding_size)

        # Running sum (over rows) of the target column's attention, and number of rows, so that the
        # average is exact whatever the number of memory chunks.
        captured = [torch.zeros(col_size, device=src.device), 0]

        @memory_chunking(num_mem_chunks)
        def feature_attention(x):
            # need_weights=False lets PyTorch use scaled_dot_product_attention, which never
            # materializes the (B*R*H, C, C) weights. When capturing (interpretability), only the
            # target column's row of the attention is computed, separately (target_attention).
            attn_output = self.self_attention_between_features(x, x, x, need_weights=False)[0]
            if self.save_feature_attention:
                captured[0] += self.target_attention(self.self_attention_between_features, x).float().sum(dim=0)
                captured[1] += x.shape[0]
            return attn_output + x

        src = feature_attention(src)
        if self.save_feature_attention:
            # target column as query attending to every column, averaged over rows and heads -> (C,)
            self.feature_attention = (captured[0] / captured[1]).detach()
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src)
        # attention between datapoints
        src = src.transpose(1, 2)
        src = src.reshape(batch_size * col_size, rows_size, embedding_size)

        @memory_chunking(num_mem_chunks)
        def datapoint_attention(x):
            # The weights are never used here, so never materialize them.
            # training data attends to itself
            x_left = self.self_attention_between_datapoints(
                x[:, :train_test_split_index],
                x[:, :train_test_split_index],
                x[:, :train_test_split_index],
                need_weights=False,
            )[0]
            # test data attends to the training data
            x_right = self.self_attention_between_datapoints(
                x[:, train_test_split_index:],
                x[:, :train_test_split_index],
                x[:, :train_test_split_index],
                need_weights=False,
            )[0]
            return torch.cat([x_left, x_right], dim=1) + x

        src = datapoint_attention(src)
        src = src.reshape(batch_size, col_size, rows_size, embedding_size)
        src = src.transpose(2, 1)
        src = self.norm2(src)
        # MLP after attention
        src = src.reshape(-1, embedding_size)

        @memory_chunking(num_mem_chunks)
        def mlp(x):
            return self.linear2(F.gelu(self.linear1(x))) + x

        src = mlp(src)
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm3(src)
        return src


def memory_chunking(num_mem_chunks: int) -> callable:
    """
    This decorator will split the first dimension of the input into chunks and apply the wrapped function
    to each chunk separately.
    Args:
        num_mem_chunks: (int) Number of chunks to split the input into, higher values use less memory but are slower.
                          Needs to be set to 1 during training to disable chunking and get correct gradients.
    """

    def decorator(func: Callable[[torch.Tensor], torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
        def wrapper(x: torch.Tensor) -> torch.Tensor:
            if num_mem_chunks <= 1 or x.shape[0] == 0:
                return func(x)
            elif torch.is_grad_enabled():
                warnings.warn(
                    "Memory chunking is disabled since gradient computation is enabled to avoid incorrect gradients. "
                    "Please use `with torch.no_grad():` during inference to enable chunking.",
                    stacklevel=2,
                )
                return func(x)
            chunk_size = max(1, math.ceil(x.shape[0] / num_mem_chunks))
            for x_split in torch.split(x, split_size_or_sections=chunk_size, dim=0):
                x_split[:] = func(
                    x_split
                )  # in-place modification to save memory, will cause wrong gradients if used during training
            return x

        return wrapper

    return decorator


class Decoder(nn.Module):
    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int):
        """Initializes the linear layers for use in the forward"""
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies an MLP to the embeddings to get the logits

        Args:
            x: (torch.Tensor) a tensor of shape (batch_size, num_rows, embedding_size)
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_outputs)
        """
        return self.linear2(F.gelu(self.linear1(x)))
