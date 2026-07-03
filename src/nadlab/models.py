from __future__ import annotations

import numpy as np
import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 48, latent_dim: int = 20, dropout: float = 0.10) -> None:
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_dim, batch_first=True, dropout=dropout if hidden_dim > 0 else 0.0)
        self.to_latent = nn.Sequential(nn.Linear(hidden_dim, latent_dim), nn.Tanh())
        self.from_latent = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU())
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(x)
        latent = self.to_latent(hidden[-1])
        repeated = self.from_latent(latent).unsqueeze(1).expand(-1, x.size(1), -1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)


class TransformerAutoencoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 48,
        latent_dim: int = 20,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.10,
        max_steps: int = 512,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads.")
        self.input_projection = nn.Linear(n_features, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, max_steps, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.to_latent = nn.Sequential(nn.Linear(hidden_dim, latent_dim), nn.Tanh())
        self.from_latent = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU())
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=layers)
        self.output = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.position.size(1):
            raise ValueError("Sequence length exceeds max_steps.")
        embedded = self.input_projection(x) + self.position[:, : x.size(1)]
        encoded = self.encoder(embedded)
        latent = self.to_latent(encoded.mean(dim=1))
        decoded_input = self.from_latent(latent).unsqueeze(1).expand(-1, x.size(1), -1)
        decoded = self.decoder(decoded_input + self.position[:, : x.size(1)])
        return self.output(decoded)


def normalized_adjacency(adjacency: np.ndarray) -> torch.Tensor:
    matrix = np.asarray(adjacency, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be a square matrix.")
    matrix = np.maximum(matrix, 0)
    matrix = matrix + np.eye(matrix.shape[0], dtype=np.float32)
    degree = np.sum(matrix, axis=1)
    inverse_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-6)))
    return torch.tensor(inverse_sqrt @ matrix @ inverse_sqrt, dtype=torch.float32)


def correlation_adjacency(train_tensor: np.ndarray) -> np.ndarray:
    """Infer an unsigned host graph from normal training telemetry only."""
    if train_tensor.ndim != 3:
        raise ValueError("Expected [time, hosts, features] tensor.")
    _, hosts, features = train_tensor.shape
    host_vectors = train_tensor.transpose(1, 0, 2).reshape(hosts, -1)
    correlation = np.corrcoef(host_vectors)
    correlation = np.nan_to_num(np.abs(correlation), nan=0.0)
    np.fill_diagonal(correlation, 0.0)
    if features == 0:
        raise ValueError("Tensor must contain features.")
    return correlation


class GraphTemporalAutoencoder(nn.Module):
    """Dependency-light graph-temporal autoencoder with learned host messages."""
    def __init__(self, n_features: int, adjacency: np.ndarray, hidden_dim: int = 48, dropout: float = 0.10) -> None:
        super().__init__()
        self.register_buffer("adjacency", normalized_adjacency(adjacency))
        self.node_projection = nn.Sequential(
            nn.Linear(n_features * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, hosts, features]
        if x.ndim != 4:
            raise ValueError("Expected graph input [batch, time, hosts, features].")
        if x.size(2) != self.adjacency.size(0):
            raise ValueError("Graph host count differs from adjacency.")
        neighbor = torch.einsum("ij,btjf->btif", self.adjacency, x)
        node_state = self.node_projection(torch.cat([x, neighbor], dim=-1))
        batch, steps, hosts, hidden = node_state.shape
        sequence = node_state.permute(0, 2, 1, 3).reshape(batch * hosts, steps, hidden)
        decoded, _ = self.temporal(sequence)
        output = self.output(decoded).reshape(batch, hosts, steps, -1).permute(0, 2, 1, 3)
        return output


def reconstruction_scores(x: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    if x.shape != reconstruction.shape:
        raise ValueError("Input and reconstruction shapes must match.")
    return torch.mean((x - reconstruction) ** 2, dim=tuple(range(1, x.ndim)))
