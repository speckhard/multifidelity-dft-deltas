"""
MegaPaiNN: PaiNN surrogate for absolute property prediction on crystals.

Adapted from DeltaPaiNN for Phase 2 (Active Learning). Key differences:
- No cheap-DFT or geometry inputs (absolute prediction, not delta-learning)
- FiLM conditioning on precision_settings only (cond_dim=12)
- Single scalar output per head (MSELoss, no logvar)
- No position/geometry heads (no denoising task)
- New embedding head for clustering/acquisition in active learning loop
- Returns dict: {e_form, e_gap, embedding}

Reuses painn_layer.py (PaiNNInteraction, FiLMGenerator, etc.) unchanged.
"""

import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool
from .painn_layer import PaiNNInteraction


class MegaPaiNN(nn.Module):

    def __init__(self,
                 num_layers: int,
                 hidden_dim: int,
                 num_precision_settings: int = 13,
                 max_z: int = 120,
                 num_rbf: int = 20,
                 cutoff: float = 5.0,
                 embedding_dim: int = 128,
                 use_film: bool = False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        cond_dim = num_precision_settings

        # --- Embeddings ---
        self.z_embedding = nn.Embedding(max_z, hidden_dim)

        # --- PaiNN Interaction Layers ---
        self.layers = nn.ModuleList([
            PaiNNInteraction(hidden_dim, num_rbf, cutoff, cond_dim,
                             use_film=use_film)
            for _ in range(num_layers)
        ])

        # --- Readout ---
        self.settings_embed = nn.Linear(num_precision_settings, hidden_dim)

        readout_dim = hidden_dim + hidden_dim  # pool + prec_embed

        # Formation energy head (extensive -> sum pool, single scalar)
        self.energy_head = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Bandgap head (intensive -> mean pool, single scalar)
        self.bandgap_head = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Embedding head (for active learning clustering/acquisition)
        self.embedding_head = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, data):
        # --- 1. Unpack ---
        pos = data.x                                         # [N, 3]
        batch = data.batch                                   # [N]
        edge_index = data.edge_index                         # [2, E]
        z = data.z.long().view(-1)                           # [N]
        global_prec = data.precision_settings                # [B, 12]

        # --- 2. Embedding ---
        s = self.z_embedding(z)                              # [N, F]
        N, F = s.size()
        v = torch.zeros(N, F, 3, device=s.device, dtype=s.dtype)

        # --- 3. Per-node conditioning (precision only) ---
        cond = global_prec[batch]                            # [N, 12]

        # --- 4. PaiNN Message Passing ---
        for layer in self.layers:
            s, v = layer(s, v, edge_index, pos, cond)

        # --- 5. Graph Pooling ---
        h_sum = global_add_pool(s, batch)                    # [B, F]
        h_mean = global_mean_pool(s, batch)                  # [B, F]
        prec_readout = self.settings_embed(global_prec)      # [B, F]

        # --- 6. Heads ---
        # Energy (extensive)
        in_energy = torch.cat([h_sum, prec_readout], dim=1)  # [B, 2F]
        e_form = self.energy_head(in_energy).squeeze(-1)     # [B]

        # Bandgap (intensive)
        in_gap = torch.cat([h_mean, prec_readout], dim=1)    # [B, 2F]
        e_gap = self.bandgap_head(in_gap).squeeze(-1)        # [B]

        # Embedding (from mean pool, no prec — pure structural)
        embedding = self.embedding_head(h_mean)              # [B, emb_dim]

        return {
            'e_form': e_form,
            'e_gap': e_gap,
            'embedding': embedding,
        }
