"""Synthetic multi-omics prior, first version (NumPy only).

Each dataset is built in two stages:

1. A small causal graph over a few dozen omics variables, organised in layers
   (SNP, CNV, MUT -> METH -> MIRNA -> RNA -> PROT). It contains the mechanism of the label
   (the variables causally linked to Y: its parents, its children and the other parents of
   its children), other mechanisms that do not involve Y, and unconnected variables. Hidden
   latent factors (batch, cell composition...) add structured variation unrelated to Y.
2. Widening, layer by layer: noisy copies of the mechanism variables (redundant signal),
   noisy copies of the other variables (structured but irrelevant) and pure-noise features,
   each mapped back to the distribution of its layer.

Every feature keeps its ground truth: its layer, whether it is discrete, and its relevance
level. Values are generated on a latent Gaussian-like scale and then mapped to each layer's
distribution (genotypes, copy-number levels, mutation status, beta values, log expression).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LAYERS = ("SNP", "CNV", "MUT", "METH", "MIRNA", "RNA", "PROT")
GENOMIC_LAYERS = ("SNP", "CNV", "MUT")  # no parents: fixed at birth
DISCRETE_LAYERS = ("SNP", "CNV", "MUT")
Y_PARENT_LAYERS = ("MUT", "METH", "MIRNA", "RNA", "PROT")  # germline SNP/CNV act through other layers
Y_CHILD_LAYERS = ("METH", "MIRNA", "RNA", "PROT")  # molecular states the label (e.g. subtype) changes
MIRNA_TARGET_LAYERS = ("RNA", "PROT")

# Relevance of each feature for the label (ground truth for explainability)
NOISE = 0  # pure noise feature
IRRELEVANT = 1  # structured, but with no causal path to or from Y (or a copy of such a variable)
CONNECTED = 2  # linked to Y beyond its Markov blanket, e.g. a distant ancestor (or a copy)
MECHANISM_COPY = 3  # noisy copy of a mechanism variable: redundant signal
MECHANISM = 4  # in the Markov blanket of Y: parents, children and co-parents of children


@dataclass
class OmicsPriorConfig:
    """Ranges from which every dataset samples its hyperparameters."""

    min_rows: int = 40
    max_rows: int = 300
    min_base_vars: int = 10
    max_base_vars: int = 60
    max_total_features: int = 2000  # after widening; 0 disables widening
    max_classes: int = 10
    max_parents: int = 3
    max_y_parents: int = 4
    prob_y_children: float = 0.5  # probability that the label has effects (children) at all
    max_y_children: int = 20
    max_latents: int = 4
    node_noise: tuple[float, float] = (0.1, 1.0)  # log-uniform, relative to the mechanism signal
    y_noise: tuple[float, float] = (0.1, 4.0)  # log-uniform temperature of the label noise
    class_effect: tuple[float, float] = (0.1, 2.0)  # log-uniform strength of the label on its children
    copy_noise: tuple[float, float] = (0.5, 3.0)  # log-uniform; >= 0.5 keeps copies worse than originals
    max_frac_noise_features: float = 0.5
    max_frac_mechanism_copies: float = 0.5
    layers: tuple[str, ...] = field(default=LAYERS)


@dataclass
class OmicsDataset:
    """A generated dataset and its ground truth. Rows are i.i.d. and in random order."""

    X: np.ndarray  # (n_rows, n_features) float
    y: np.ndarray  # (n_rows,) int, classes 0..num_classes-1
    layer: np.ndarray  # (n_features,) layer name of each feature
    is_discrete: np.ndarray  # (n_features,) bool
    relevance: np.ndarray  # (n_features,) relevance level (NOISE ... MECHANISM)
    is_base: np.ndarray  # (n_features,) True for the causal-graph variables, False for widened ones
    num_classes: int


def _loguniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _standardize(a):
    """Column-wise standardization; constant columns become 0."""
    a = np.asarray(a, dtype=float)
    std = a.std(axis=0)
    return (a - a.mean(axis=0)) / np.where(std > 0, std, 1.0)


def _mechanism(rng, parents):
    """A random function of the (standardized) parents, standardized: linear, small MLP or threshold."""
    n, k = parents.shape
    kind = rng.choice(["linear", "mlp", "threshold"], p=[0.4, 0.4, 0.2])
    if kind == "linear":
        out = parents @ rng.normal(0, 1, k)
    elif kind == "mlp":
        hidden = np.tanh(parents @ rng.normal(0, 1 / np.sqrt(k), (k, 8)) + rng.normal(0, 0.5, 8))
        out = hidden @ rng.normal(0, 1, 8)
    else:  # regulation-like on/off switches
        out = (parents > rng.normal(0, 0.7, k)).astype(float) @ rng.normal(0, 1, k)
    return _standardize(out[:, None])[:, 0]


def _ranks(z):
    """Column-wise ranks in (0, 1), so that thresholds can be set by quantile."""
    n = z.shape[0]
    return (np.argsort(np.argsort(z, axis=0), axis=0) + 0.5) / n


def _map_to_layer(rng, z, layer):
    """Maps latent values z of shape (n, m) to the observed distribution of the layer. Each
    column draws its own parameters (allele frequency, mutation rate, ...)."""
    n, m = z.shape
    if layer == "SNP":  # genotypes 0/1/2 in Hardy-Weinberg proportions
        maf = rng.uniform(0.05, 0.5, m)
        q0, q1 = (1 - maf) ** 2, 2 * maf * (1 - maf)
        u = _ranks(z)
        return (u > q0).astype(float) + (u > q0 + q1).astype(float)
    if layer == "CNV":  # copy-number levels -2..2, mostly 0
        p_zero = rng.uniform(0.5, 0.9, m)
        tails = rng.dirichlet(np.ones(4), m) * (1 - p_zero)[:, None]  # -2, -1, +1, +2
        cuts = np.cumsum(np.column_stack([tails[:, :2], p_zero, tails[:, 2:3]]), axis=1)  # (m, 4)
        u = _ranks(z)
        return (u[:, :, None] > cuts[None, :, :]).sum(axis=2).astype(float) - 2.0
    if layer == "MUT":  # somatic mutation status, sparse
        rate = np.exp(rng.uniform(np.log(0.02), np.log(0.3), m))
        return (_ranks(z) > 1 - rate).astype(float)
    if layer == "METH":  # beta values in (0, 1), often bimodal
        slope = rng.uniform(1.0, 4.0, m)
        shift = rng.normal(0, 1.5, m)
        return 1 / (1 + np.exp(-(slope * _standardize(z) + shift)))
    if layer == "MIRNA":  # log expression with a floor (not detected)
        x = _standardize(z)
        floor = np.quantile(x, rng.uniform(0, 0.5, m), axis=0).diagonal() if m else np.zeros(0)
        return np.maximum(x, floor)
    return _standardize(z)  # RNA, PROT: log expression


def _sample_roots(rng, n, sampling):
    if sampling == "normal":
        return rng.normal(0, 1, n)
    if sampling == "uniform":
        return rng.uniform(-np.sqrt(3), np.sqrt(3), n)
    return np.where(rng.random(n) < 0.5, rng.normal(-1, 0.5, n), rng.normal(1, 0.5, n))  # bimodal


def _ancestors(adj, node):
    """All ancestors of node in the adjacency matrix adj[parent, child]."""
    seen, stack = set(), [node]
    while stack:
        for p in np.flatnonzero(adj[:, stack.pop()]):
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen


def _descendants(adj, node):
    return _ancestors(adj.T, node)


def _topological_order(adj):
    indegree = adj.sum(axis=0).astype(int)
    order, ready = [], list(np.flatnonzero(indegree == 0))
    while ready:
        node = ready.pop()
        order.append(node)
        for child in np.flatnonzero(adj[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    assert len(order) == adj.shape[0], "graph has a cycle"
    return order


def _sample_graph(rng, config):
    """Samples the layered causal graph. Returns (node layers, adjacency over observed variables,
    latent factors and the label, index of the label node, number of latents)."""
    for _ in range(100):
        n_base = int(rng.integers(config.min_base_vars, config.max_base_vars + 1))
        n_active = int(rng.integers(2, len(config.layers) + 1))
        active = [layer for layer in config.layers if layer in rng.choice(config.layers, n_active, replace=False)]
        if not any(layer in Y_PARENT_LAYERS for layer in active):
            continue
        counts = 1 + rng.multinomial(n_base - len(active), rng.dirichlet(np.full(len(active), rng.uniform(0.3, 3.0))))
        layers = np.array([layer for layer, c in zip(active, counts) for _ in range(c)])
        rank = {layer: i for i, layer in enumerate(config.layers)}

        n_latents = int(rng.integers(0, config.max_latents + 1))
        n_nodes = n_base + n_latents + 1  # observed, latent factors, label
        y_node = n_nodes - 1
        adj = np.zeros((n_nodes, n_nodes), dtype=bool)

        # Edges among observed variables, following the layer order
        max_parents = int(rng.integers(1, config.max_parents + 1))
        weights = {"intra": rng.uniform(0.5, 2), "adjacent": rng.uniform(0.5, 2), "skip": rng.uniform(0.1, 1)}
        for child in range(n_base):
            if layers[child] in GENOMIC_LAYERS:
                continue
            candidates, w = [], []
            for parent in range(child):
                lp, lc = layers[parent], layers[child]
                if lp == "MIRNA" and lc not in MIRNA_TARGET_LAYERS:
                    continue
                gap = rank[lc] - rank[lp]
                candidates.append(parent)
                w.append(weights["intra"] if gap == 0 else weights["adjacent"] if gap == 1 else weights["skip"])
            k = min(int(rng.integers(0, max_parents + 1)), len(candidates))
            if k:
                chosen = rng.choice(candidates, k, replace=False, p=np.array(w) / np.sum(w))
                adj[chosen, child] = True

        # Latent factors: structured variation shared by many variables, unrelated to the label
        molecular = np.flatnonzero(~np.isin(layers, GENOMIC_LAYERS))
        for latent in range(n_base, n_base + n_latents):
            if len(molecular):
                size = max(1, int(rng.uniform(0.1, 0.5) * len(molecular)))
                adj[latent, rng.choice(molecular, size, replace=False)] = True

        # Parents of the label
        parent_pool = np.flatnonzero(np.isin(layers, Y_PARENT_LAYERS))
        if not len(parent_pool):
            continue
        n_y_parents = min(int(rng.integers(1, config.max_y_parents + 1)), len(parent_pool))
        y_parents = rng.choice(parent_pool, n_y_parents, replace=False)
        adj[y_parents, y_node] = True

        # Children of the label: states it changes, which must not be ancestors of its parents
        if rng.random() < config.prob_y_children:
            forbidden = set(y_parents.tolist())
            for p in y_parents:
                forbidden |= _ancestors(adj, p)
            child_pool = [
                i for i in np.flatnonzero(np.isin(layers, Y_CHILD_LAYERS)) if i not in forbidden
            ]
            if child_pool:
                n_children = min(int(rng.integers(1, config.max_y_children + 1)), len(child_pool))
                adj[y_node, rng.choice(child_pool, n_children, replace=False)] = True
        return layers, adj, y_node, n_latents
    raise RuntimeError("could not sample a valid graph")


def _relevance_of_base(adj, y_node, n_base):
    """Relevance level of each observed variable from the graph."""
    parents = set(np.flatnonzero(adj[:, y_node]).tolist())
    children = set(np.flatnonzero(adj[y_node]).tolist())
    co_parents = set()
    for c in children:
        co_parents |= set(np.flatnonzero(adj[:, c]).tolist())
    blanket = (parents | children | co_parents) - {y_node}
    connected = (_ancestors(adj, y_node) | _descendants(adj, y_node)) - blanket - {y_node}
    relevance = np.full(n_base, IRRELEVANT)
    for i in range(n_base):
        if i in blanket:
            relevance[i] = MECHANISM
        elif i in connected:
            relevance[i] = CONNECTED
    return relevance


def _generate_values(rng, config, layers, adj, y_node, n_latents, n_rows):
    """Samples latent and observed values in topological order. Returns (observed X, latent Z of
    the observed variables, y, number of classes)."""
    n_base = len(layers)
    n_nodes = adj.shape[0]
    num_classes = 2 if rng.random() < 0.5 else int(rng.integers(3, config.max_classes + 1))
    sampling = rng.choice(["normal", "uniform", "bimodal"])
    base_noise = _loguniform(rng, *config.node_noise)
    y_temperature = _loguniform(rng, *config.y_noise)
    class_effect = _loguniform(rng, *config.class_effect)

    latent = np.zeros((n_rows, n_nodes))  # latent scale of every node
    observed = np.zeros((n_rows, n_base))  # observed values of the observed variables
    y = np.zeros(n_rows, dtype=int)
    y_onehot = np.zeros((n_rows, num_classes))

    def parent_matrix(node):
        cols = []
        for p in np.flatnonzero(adj[:, node]):
            if p == y_node:
                cols.append(y_onehot * class_effect)  # class effects on the label's children
            elif p < n_base:
                cols.append(_standardize(observed[:, [p]]))
            else:
                cols.append(latent[:, [p]])
        return np.column_stack(cols) if cols else None

    for node in _topological_order(adj):
        parents = parent_matrix(node)
        if node == y_node:
            logits = np.column_stack([_mechanism(rng, parents) for _ in range(num_classes)])
            logits = logits + rng.normal(0, 0.5, num_classes)  # class biases -> imbalance
            y = np.argmax(logits + y_temperature * rng.gumbel(size=logits.shape), axis=1)
            y_onehot = np.eye(num_classes)[y]
            continue
        if parents is None:
            z = _sample_roots(rng, n_rows, sampling)
        else:
            noise = base_noise * rng.uniform(0.5, 1.5)
            z = _mechanism(rng, parents) + noise * rng.normal(0, 1, n_rows)
        latent[:, node] = z
        if node < n_base:
            observed[:, node] = _map_to_layer(rng, z[:, None], layers[node])[:, 0]
    return observed, latent[:, :n_base], y, num_classes


def _widen(rng, config, layers, z_base, relevance, n_rows):
    """Adds features layer by layer: noisy copies of mechanism variables, noisy copies of the
    other variables and pure noise, each mapped to its layer's distribution."""
    n_base = len(layers)
    if config.max_total_features <= n_base:
        return np.zeros((n_rows, 0)), np.array([], dtype=layers.dtype), np.zeros(0, dtype=int)
    n_total = int(_loguniform(rng, n_base, config.max_total_features))
    n_new = n_total - n_base
    active = list(dict.fromkeys(layers.tolist()))
    counts = rng.multinomial(n_new, rng.dirichlet(np.array([np.sum(layers == la) for la in active], float)))
    frac_noise = rng.uniform(0, config.max_frac_noise_features)
    frac_mech = rng.uniform(0, config.max_frac_mechanism_copies)
    copy_noise = _loguniform(rng, *config.copy_noise)

    blocks, new_layers, new_relevance = [], [], []
    for layer, count in zip(active, counts):
        if count == 0:
            continue
        idx = np.flatnonzero(layers == layer)
        mech = idx[relevance[idx] == MECHANISM]
        other = idx[relevance[idx] != MECHANISM]
        n_noise = int(round(frac_noise * count))
        n_mech = int(round(frac_mech * (count - n_noise))) if len(mech) else 0
        n_other = count - n_noise - n_mech
        if not len(other):  # the whole layer is mechanism: its remaining copies are mechanism copies
            n_mech, n_other = n_mech + n_other, 0

        latent_cols, levels = [rng.normal(0, 1, (n_rows, n_noise))], [np.full(n_noise, NOISE)]
        for pool, n_copies in ((mech, n_mech), (other, n_other)):
            if n_copies == 0:
                continue
            sources = _standardize(z_base[:, pool])
            k = rng.integers(1, min(2, len(pool)) + 1, n_copies)  # 1 or 2 sources per copy
            weights = np.zeros((len(pool), n_copies))
            src_levels = np.zeros(n_copies, dtype=int)
            for j in range(n_copies):
                chosen = rng.choice(len(pool), k[j], replace=False)
                weights[chosen, j] = rng.normal(0, 1, k[j])
                top = relevance[pool[chosen]].max()
                src_levels[j] = MECHANISM_COPY if top == MECHANISM else top
            signal = _standardize(sources @ weights)
            latent_cols.append(signal + copy_noise * rng.normal(0, 1, signal.shape))
            levels.append(src_levels)
        z_new = np.column_stack(latent_cols)
        blocks.append(_map_to_layer(rng, z_new, layer))
        new_layers.append(np.full(z_new.shape[1], layer))
        new_relevance.append(np.concatenate(levels))
    if not blocks:  # the sampled width left nothing to add
        return np.zeros((n_rows, 0)), np.array([], dtype=layers.dtype), np.zeros(0, dtype=int)
    return np.column_stack(blocks), np.concatenate(new_layers), np.concatenate(new_relevance)


def sample_dataset(rng: np.random.Generator, config: OmicsPriorConfig | None = None) -> OmicsDataset:
    """Samples one synthetic multi-omics classification dataset with its ground truth."""
    config = config or OmicsPriorConfig()
    for _ in range(100):
        n_rows = int(_loguniform(rng, config.min_rows, config.max_rows))
        layers, adj, y_node, n_latents = _sample_graph(rng, config)
        observed, z_base, y, num_classes = _generate_values(rng, config, layers, adj, y_node, n_latents, n_rows)
        if np.bincount(y, minlength=num_classes).min() < 2:  # every class needs a few samples
            continue
        relevance = _relevance_of_base(adj, y_node, len(layers))
        X_new, new_layers, new_relevance = _widen(rng, config, layers, z_base, relevance, n_rows)
        X = np.column_stack([observed, X_new])
        layer = np.concatenate([layers, new_layers])
        rel = np.concatenate([relevance, new_relevance])
        is_base = np.concatenate([np.ones(len(layers), bool), np.zeros(len(new_layers), bool)])
        perm = rng.permutation(X.shape[1])
        return OmicsDataset(
            X=X[:, perm],
            y=y,
            layer=layer[perm],
            is_discrete=np.isin(layer[perm], DISCRETE_LAYERS),
            relevance=rel[perm],
            is_base=is_base[perm],
            num_classes=num_classes,
        )
    raise RuntimeError("could not sample a dataset with every class present")
