import numpy as np
import torch

from nadlab.models import GraphTemporalAutoencoder, LSTMAutoencoder, TransformerAutoencoder, reconstruction_scores


def test_sequence_autoencoders_preserve_input_shape():
    x = torch.randn(5, 12, 8)
    lstm = LSTMAutoencoder(n_features=8, hidden_dim=16, latent_dim=6)
    transformer = TransformerAutoencoder(n_features=8, hidden_dim=16, latent_dim=6, heads=4, layers=1)
    assert lstm(x).shape == x.shape
    assert transformer(x).shape == x.shape
    assert reconstruction_scores(x, lstm(x)).shape == (5,)


def test_graph_temporal_autoencoder_preserves_input_shape():
    x = torch.randn(4, 10, 3, 8)
    adjacency = np.ones((3, 3), dtype=np.float32) - np.eye(3, dtype=np.float32)
    model = GraphTemporalAutoencoder(n_features=8, adjacency=adjacency, hidden_dim=16)
    assert model(x).shape == x.shape
