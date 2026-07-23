"""Utility for encoding column names into text embeddings using a frozen sentence encoder.

This module lives OUTSIDE the model. The model only receives pre-computed
embedding tensors - it never imports or depends on sentence-transformers.
"""

import torch


class ColumnTextEncoder:
    """Encodes column name strings into fixed-size vectors using a frozen sentence encoder.

    Usage in prior/dataloader (batched):
        encoder = ColumnTextEncoder()
        batch_embs = []
        for table_column_names in batch_of_tables:
            batch_embs.append(encoder.encode(table_column_names))  # (F, D)
        column_embeddings = torch.stack(batch_embs)  # (B, F, D)

    Usage in inference interface (single table):
        encoder = ColumnTextEncoder()
        column_embeddings = encoder.encode(column_names)  # (F, D)
        model(..., column_embeddings=column_embeddings)
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu'):
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(model_name, device=device)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.text_embedding_dim = self.encoder.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encode column names into embedding vectors.

        Args:
            texts: List of column name strings.
        Returns:
            (len(texts), text_embedding_dim) tensor, detached and on encoder device.
        """
        with torch.no_grad():
            embeddings = self.encoder.encode(texts, convert_to_tensor=True)
        return embeddings.detach()
