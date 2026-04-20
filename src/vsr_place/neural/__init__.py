"""NeuralVSR: Learned Graph Neural Network for amortized placement repair.

Replaces hand-crafted repulsive force (in renoising/local_repair.py) with
a trained GNN that predicts per-macro displacement given:
- current placement
- structured violation feedback from verifier
- netlist graph (edge_index, edge_features)

Trained on synthetic violations, zero-shot generalizes to real circuits.
"""

from vsr_place.neural.model import NeuralVSR
from vsr_place.neural.infer import neural_repair_loop

__all__ = ["NeuralVSR", "neural_repair_loop"]
