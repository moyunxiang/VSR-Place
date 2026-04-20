"""NeuralVSR GNN architecture.

Predicts per-macro displacement given current placement + violation
feedback. Trained on synthetic violations, applied iteratively at
inference.

Architecture:
- Input: (centers, node_features, edge_index, edge_attr)
  - centers: (N, 2) macro center positions
  - node_features: (N, F_n) per-macro features (size, violation severity, ...)
  - edge_index: (2, E) netlist graph edges
  - edge_attr: (E, F_e) edge features (pin offsets)
- Output: (N, 2) displacement per macro

~50K parameters, fits comfortably in modest GPU memory.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralVSR(nn.Module):
    """Graph Neural Network for learned placement repair.

    Uses attention-based message passing (GATv2-style) with residual
    connections. Output layer initialized to zero so initial displacement
    is small (stable iterative repair).

    Args:
        node_feat_dim: Number of per-macro input features (excluding position).
            Default: 5 = [severity, boundary, overlap_count, width, height].
        edge_feat_dim: Number of per-edge features. Default: 4 = pin offsets.
        hidden_dim: Hidden layer width.
        num_layers: Number of GAT blocks.
        heads: Attention heads per layer.
        dropout: Dropout rate in GAT layers.
    """

    def __init__(
        self,
        node_feat_dim: int = 5,
        edge_feat_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        from torch_geometric.nn import GATv2Conv

        # Position (2) + node features = input dim
        in_dim = 2 + node_feat_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        per_head = hidden_dim // heads
        assert per_head * heads == hidden_dim, "hidden_dim must be divisible by heads"

        self.gat_layers = nn.ModuleList([
            GATv2Conv(
                hidden_dim, per_head,
                heads=heads, concat=True,
                edge_dim=edge_feat_dim,
                dropout=dropout,
                add_self_loops=True,
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, 2)
        # Initialize output to zero → initial displacement is zero
        # (stable for iterative repair: first forward pass doesn't move anything much)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(
        self,
        centers: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict per-macro displacement.

        Args:
            centers: (N, 2) macro centers.
            node_features: (N, node_feat_dim) per-macro features.
            edge_index: (2, E) edge connectivity (undirected/bidirectional).
            edge_attr: (E, edge_feat_dim) edge features, or None.

        Returns:
            (N, 2) predicted displacement.
        """
        # Concatenate position and features as initial node representation
        h = torch.cat([centers, node_features], dim=-1)  # (N, 2+F)
        h = self.input_proj(h)
        h = F.relu(h)

        for gat, norm in zip(self.gat_layers, self.norms):
            h_res = h
            h = gat(h, edge_index, edge_attr=edge_attr)
            h = F.relu(h)
            h = norm(h + h_res)

        delta = self.output_proj(h)
        return delta

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
