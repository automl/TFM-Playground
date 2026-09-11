import torch
from tabfm.src.pytorch.model import MLP, TabFM
from torch import nn

from tfmplayground.configs.models import TabFMClassifierConfig, TabFMRegressorConfig
from tfmplayground.models.base import TabularFoundationModel


class TabFMModel(TabFM, TabularFoundationModel):
    """
    adapts tabfm to tabularfoundationmodel interface
    """

    def __init__(self, config: TabFMClassifierConfig | TabFMRegressorConfig) -> None:
        """
        builds tabfm from config and makes its fourier frequencies random
        """
        self.config = config
        super().__init__(
            embed_dim=config.embed_dim,
            max_classes=config.max_classes,
            col_num_blocks=config.col_num_blocks,
            col_nhead=config.col_nhead,
            col_num_inds=config.col_num_inds,
            row_num_blocks=config.row_num_blocks,
            row_nhead=config.row_nhead,
            row_num_cls=config.row_num_cls,
            icl_num_blocks=config.icl_num_blocks,
            icl_nhead=config.icl_nhead,
            ff_factor=config.ff_factor,
            feature_group_size=config.feature_group_size,
            num_freq=config.num_freq,
            decoder_hidden=config.decoder_hidden,
            is_classifier=config.is_classifier,
        )
        if not config.is_classifier:
            self.rebuild_decoder(config.decoder_out)
            self.register_buffer("borders", torch.zeros(config.decoder_out + 1))
        fourier_sigma = config.fourier_sigma
        for name in ("fourier_frequencies", "fourier_frequencies_cat"):
            # no jax checkpoint fills these so we make them random parameters
            zeros = getattr(self.cell_embedder, name)
            delattr(self.cell_embedder, name)
            frequencies = torch.randn_like(zeros) * fourier_sigma
            self.cell_embedder.register_parameter(name, nn.Parameter(frequencies))

    def rebuild_decoder(self, decoder_out: int) -> None:
        """
        rebuilds decoder for specified output size
        """
        decoder = self.icl_predictor.decoder
        if decoder_out == decoder.layers[-1].out_features:
            return
        in_dim = decoder.layers[0].in_features
        hidden_dim = decoder.layers[0].out_features
        self.icl_predictor.decoder = MLP(in_dim, [hidden_dim], decoder_out)

    def forward(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
    ) -> torch.Tensor:
        """
        takes train rows as context and predicts test rows through tabfm forward

        joins train and test rows, pads targets with zeros, and slices test outputs
        """
        batch_size, _ = y_train.shape
        _, num_train_rows, _ = X_train.shape
        _, num_test_rows, _ = X_test.shape
        train_size = torch.full([batch_size], num_train_rows, device=X_train.device, dtype=torch.long)
        y_test_zeros = torch.zeros(batch_size, num_test_rows, device=y_train.device, dtype=y_train.dtype)
        x = torch.cat([X_train, X_test], dim=1)
        y = torch.cat([y_train, y_test_zeros], dim=1)
        output = super().forward(x, y, train_size)
        return output[:, num_train_rows:]
