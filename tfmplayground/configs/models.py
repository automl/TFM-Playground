from dataclasses import dataclass


@dataclass
class NanoTabPFNModelConfig:
    """
    settings nanotabpfn models share
    """

    embedding_size: int = 192
    num_attention_heads: int = 6
    mlp_hidden_size: int = 768
    num_layers: int = 6
    num_mem_chunks: int = 1


@dataclass
class NanoTabPFNClassifierConfig(NanoTabPFNModelConfig):
    """
    nanotabpfn settings for classification
    """

    problem: str = "classification"
    num_outputs: int = 10


@dataclass
class NanoTabPFNRegressorConfig(NanoTabPFNModelConfig):
    """
    nanotabpfn settings for regression
    """

    problem: str = "regression"
    head: str = "buckets"
    num_outputs: int = 1000


@dataclass
class TabICLModelConfig:
    """
    settings tabicl models share
    """

    num_quantiles: int = 999
    embed_dim: int = 128
    col_num_blocks: int = 3
    col_nhead: int = 8
    col_num_inds: int = 128
    col_affine: bool = False
    col_feature_group: str = "same"
    col_feature_group_size: int = 3
    col_target_aware: bool = True
    col_ssmax: str = "qassmax-mlp-elementwise"
    row_num_blocks: int = 3
    row_nhead: int = 8
    row_num_cls: int = 4
    row_rope_base: float = 100000
    row_rope_interleaved: bool = True
    icl_num_blocks: int = 12
    icl_nhead: int = 8
    icl_ssmax: str = "qassmax-mlp-elementwise"
    ff_factor: int = 2
    dropout: float = 0.0
    activation: str = "gelu"
    norm_first: bool = True
    bias_free_ln: bool = False
    zero_init: bool = True
    recompute: bool = False


@dataclass
class TabICLClassifierConfig(TabICLModelConfig):
    """
    tabicl settings for classification
    """

    problem: str = "classification"
    max_classes: int = 10


@dataclass
class TabICLRegressorConfig(TabICLModelConfig):
    """
    tabicl settings for regression
    """

    problem: str = "regression"
    head: str = "quantiles"
    max_classes: int = 0


@dataclass
class NanoTabICLModelConfig:
    """
    settings nanotabicl models share
    """

    embed_dim: int = 128
    col_num_blocks: int = 3
    row_num_blocks: int = 3
    icl_num_blocks: int = 12
    col_nhead: int = 8
    row_nhead: int = 8
    icl_nhead: int = 8
    feature_group_size: int = 3
    n_cls_cols: int = 4
    n_cls_rows: int = 128


@dataclass
class NanoTabICLClassifierConfig(NanoTabICLModelConfig):
    """
    nanotabicl settings for classification
    """

    problem: str = "classification"
    max_classes: int = 10
    out_dim: int = 10


@dataclass
class NanoTabICLRegressorConfig(NanoTabICLModelConfig):
    """
    nanotabicl settings for regression
    """

    problem: str = "regression"
    head: str = "quantiles"
    max_classes: int = 0
    out_dim: int = 999


@dataclass
class ModdedNanoTabPFNModelConfig:
    """
    settings moddednanotabpfn models share

    Attributes
    ----------
    l : int
        number of transformer layers
    a : int
        number of attention heads
    e : int
        embedding size
    h : int
        mlp hidden size
    residual_decay : float
        decay of residual stream per layer, 1.0 gives no decay
    thinking_rows : int
        number of learned rows added to each table
    feature_group_size : int
        number of features embedded together
    """

    l: int = 5
    a: int = 4
    e: int = 256
    h: int = 768
    residual_decay: float = 0.95
    thinking_rows: int = 24
    feature_group_size: int = 5


@dataclass
class ModdedNanoTabPFNClassifierConfig(ModdedNanoTabPFNModelConfig):
    """
    moddednanotabpfn settings for classification
    """

    problem: str = "classification"
    o: int = 10


@dataclass
class ModdedNanoTabPFNRegressorConfig(ModdedNanoTabPFNModelConfig):
    """
    moddednanotabpfn settings for regression
    """

    problem: str = "regression"
    head: str = "buckets"
    o: int = 999


@dataclass
class TabFMModelConfig:
    """
    settings tabfm models share
    """

    fourier_sigma: float = 1.0
    embed_dim: int = 8
    max_classes: int = 3
    col_num_blocks: int = 2
    col_nhead: int = 2
    col_num_inds: int = 4
    row_num_blocks: int = 2
    row_nhead: int = 2
    row_num_cls: int = 2
    icl_num_blocks: int = 2
    icl_nhead: int = 2
    ff_factor: int = 2
    feature_group_size: int = 3
    num_freq: int = 32
    decoder_hidden: int | None = None


@dataclass
class TabFMClassifierConfig(TabFMModelConfig):
    """
    tabfm settings for classification
    """

    problem: str = "classification"
    is_classifier: bool = True


@dataclass
class TabFMRegressorConfig(TabFMModelConfig):
    """
    tabfm settings for regression
    """

    problem: str = "regression"
    head: str = "scalar"
    is_classifier: bool = False
