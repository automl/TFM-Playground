import math
import warnings
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.transformer import LayerNorm, Linear, MultiheadAttention

from tfmplayground.configs.models import NanoTabPFNClassifierConfig, NanoTabPFNRegressorConfig
from tfmplayground.models.base import TabularFoundationModel


class NanoTabPFN(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        num_attention_heads: int,
        mlp_hidden_size: int,
        num_layers: int,
        num_outputs: int,
        num_mem_chunks: int = 1,
    ):
        """Initializes the feature/target encoder, transformer blocks and decoder"""
        super().__init__()
        self.embedding_size = embedding_size
        self.num_attention_heads = num_attention_heads
        self.mlp_hidden_size = mlp_hidden_size
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.num_mem_chunks = num_mem_chunks
        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = TargetEncoder(embedding_size)
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(
                TransformerEncoderLayer(embedding_size, num_attention_heads, mlp_hidden_size)
            )
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)

    def forward(self, src: tuple[torch.Tensor, torch.Tensor], train_test_split_index: int) -> torch.Tensor:
        """
        Predicts the outputs for X_test given the labelled (X_train, y_train) context.

        Parameters
        ----------
        src : tuple of torch.Tensor
            a tensor of shape (batch_size, num_rows, num_features) that holds X_train and X_test,
            and a tensor of shape (batch_size, num_train_datapoints, 1) that holds y_train
        train_test_split_index : int
            the number of datapoints in X_train

        Returns
        -------
        torch.Tensor
            a tensor of shape (batch_size, num_test_datapoints, num_classes), which represent the predicted logits
        """
        x_src, y_src = src
        # we expect the labels to look like (batches, num_train_datapoints, 1),
        # so we add the last dimension if it is missing
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
        # from here on B=Batches, R=Rows, C=Columns, E=embedding size
        # converts scalar values to embeddings, so (B,R,C-1) -> (B,R,C-1,E)
        x_src = self.feature_encoder(x_src, train_test_split_index)
        num_rows = x_src.shape[1]
        # padds the y_train up to y by using the mean,
        # then converts scalar values to embeddings (B,R,1,E)
        y_src = self.target_encoder(y_src, num_rows)
        # concatenates the feature embeddings with the target embeddings
        # to give us the full table of embeddings (B,R,C,E))
        src = torch.cat([x_src, y_src], 2)
        # repeatedly applies the transformer block on (B,R,C,E)
        for block in self.transformer_blocks:
            src = block(
                src,
                train_test_split_index=train_test_split_index,
                num_mem_chunks=self.num_mem_chunks,
            )
        # selects the target embeddings (B,num_targets,1,E)
        output = src[:, train_test_split_index:, -1, :]
        # runs the embeddings through the decoder to get
        # the logits of our predictions (B,num_targets,num_classes)
        output = self.decoder(output)
        return output


class FeatureEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        """Creates the linear layer that we will use to embed our features."""
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, x: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        """
        Normalizes all the features based on the mean and std of the features of the training data,
        clips them between -100 and 100, then applies a linear layer to embed the features.

        Args:
            x: (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features)
            train_test_split_index: (int) the number of datapoints in X_train
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_features, embedding_size), representing
                           the embeddings of the features
        """
        x = x.unsqueeze(-1)
        mean = torch.mean(x[:, :train_test_split_index], dim=1, keepdims=True)
        std = torch.std(x[:, :train_test_split_index], dim=1, keepdims=True) + 1e-8  # TODO: maybe change the constant
        x = (x - mean) / std
        x = torch.clip(x, min=-100, max=100)
        return self.linear_layer(x)


class TargetEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        """Creates the linear layer that we will use to embed our targets."""
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, y_train: torch.Tensor, num_rows: int) -> torch.Tensor:
        """
        Pads up y_train to the full length of y using the mean per dataset and then embeds it using a linear layer

        Args:
            y_train: (torch.Tensor) a tensor of shape (batch_size, num_train_datapoints, 1)
            num_rows: (int) the full length of y
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows, 1, embedding_size), representing
                           the embeddings of the targets
        """
        # nan padding & nan handler instead?
        mean = torch.mean(y_train, axis=1, keepdim=True)
        padding = mean.repeat(1, num_rows - y_train.shape[1], 1)
        y = torch.cat([y_train, padding], dim=1)
        y = y.unsqueeze(-1)
        return self.linear_layer(y)


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

    def forward(self, src: torch.Tensor, train_test_split_index: int, num_mem_chunks: int = 1) -> torch.Tensor:
        """
        Takes the embeddings of the table as input and applies self-attention between features
        and self-attention between datapoints followed by a simple 2 layer MLP.

        Parameters
        ----------
        src : torch.Tensor
            a tensor of shape (batch_size, num_rows, num_features, embedding_size)
            that contains all the embeddings for all the cells in the table
        train_test_split_index : int
            the length of X_train
        num_mem_chunks : int
            number of chunks that memory-intense operations will be split into,
            higher values use less memory but are slower,
            needs to be set to 1 during training to get correct gradients

        Returns
        -------
        torch.Tensor
            a tensor of shape (batch_size, num_rows, num_features, embedding_size)
        """
        batch_size, rows_size, col_size, embedding_size = src.shape
        # attention between features
        src = src.reshape(batch_size * rows_size, col_size, embedding_size)

        @memory_chunking(num_mem_chunks)
        def feature_attention(x):
            return self.self_attention_between_features(x, x, x)[0] + x

        src = feature_attention(src)
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src)
        # attention between datapoints
        src = src.transpose(1, 2)
        src = src.reshape(batch_size * col_size, rows_size, embedding_size)

        @memory_chunking(num_mem_chunks)
        def datapoint_attention(x):
            # training data attends to itself
            x_left = self.self_attention_between_datapoints(
                x[:, :train_test_split_index],
                x[:, :train_test_split_index],
                x[:, :train_test_split_index],
            )[0]
            # test data attends to the training data
            x_right = self.self_attention_between_datapoints(
                x[:, train_test_split_index:],
                x[:, :train_test_split_index],
                x[:, :train_test_split_index],
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
    This decorator will split the first dimension of the input into chunks
    and apply the wrapped function to each chunk separately.

    Parameters
    ----------
    num_mem_chunks : int
        number of chunks to split the input into,
        higher values use less memory but are slower,
        needs to be set to 1 during training to disable chunking and get correct gradients

    Returns
    -------
    callable
        decorator that applies wrapped function to one chunk at a time
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


class NanoTabPFNModel(NanoTabPFN, TabularFoundationModel):
    """
    adapts nanotabpfn to tabularfoundationmodel interface
    """

    def __init__(self, config: NanoTabPFNClassifierConfig | NanoTabPFNRegressorConfig) -> None:
        """
        builds nanotabpfn from config and reserves its borders
        """
        self.config = config
        super().__init__(
            embedding_size=config.embedding_size,
            num_attention_heads=config.num_attention_heads,
            mlp_hidden_size=config.mlp_hidden_size,
            num_layers=config.num_layers,
            num_outputs=config.num_outputs,
            num_mem_chunks=config.num_mem_chunks,
        )
        self.register_buffer("borders", torch.zeros(config.num_outputs + 1))

    def forward(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> torch.Tensor:
        """
        takes train rows as context and predicts test rows through nanotabpfn forward

        joins train and test rows into one table and takes split index from train targets
        """
        src = torch.cat([X_train, X_test], dim=1), y_train
        train_test_split_index = y_train.shape[1]
        return super().forward(src, train_test_split_index)
