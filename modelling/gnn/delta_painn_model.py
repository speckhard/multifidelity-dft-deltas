"""
DeltaPaiNN: FiLM-conditioned PaiNN for delta-learning of DFT properties.

Drop-in replacement for DeltaGNN with identical forward(data) signature
and return format: ((pos_mean, pos_logvar), (e_mean, e_logvar),
                    (gap_mean, gap_logvar), (geo_mean, geo_logvar))

Key differences from EGNN models:
- Separate scalar/vector tracks prevent error propagation
- Positions are NOT updated through message passing
- Position deltas come from vector readout
- FiLM conditioning modulates scalar track per fidelity level
"""

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool, global_add_pool
from .painn_layer import PaiNNInteraction


class DeltaPaiNN(nn.Module):

    def __init__(self,
                 num_layers: int,
                 hidden_dim: int,
                 num_cheap_dft_inputs: int = 12,
                 num_precision_settings: int = 13,
                 num_geo_inputs: int = 7,
                 max_z: int = 120,
                 num_rbf: int = 20,
                 cutoff: float = 5.0,
                 use_film: bool = True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        self.num_cheap_dft_inputs = num_cheap_dft_inputs
        self.num_geo_inputs = num_geo_inputs
        cond_dim = num_cheap_dft_inputs + num_precision_settings

        # --- Embeddings ---
        self.z_embedding = nn.Embedding(max_z, hidden_dim)

        # --- PaiNN Interaction Layers ---
        self.layers = nn.ModuleList([
            PaiNNInteraction(hidden_dim, num_rbf, cutoff, cond_dim,
                             use_film=use_film)
            for _ in range(num_layers)
        ])

        # --- Readout (matches DeltaGNN pattern) ---
        self.settings_embed = nn.Linear(num_precision_settings, hidden_dim)

        base_readout_size = hidden_dim + hidden_dim + num_cheap_dft_inputs

        # 1. Delta Energy Head (mean, logvar)
        self.energy_head = nn.Sequential(
            nn.Linear(base_readout_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)
        )

        # 2. Delta Bandgap Head (mean, logvar)
        self.bandgap_head = nn.Sequential(
            nn.Linear(base_readout_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)
        )

        # 3. Geometry Head (7 means + 7 logvars)
        geo_input_size = base_readout_size + num_geo_inputs
        self.geometry_head = nn.Sequential(
            nn.Linear(geo_input_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 7 * 2)  # Always 7 geo targets (vol + 6 lattice)
        )

        # 4. Position Uncertainty Head (per-atom logvar from scalars)
        self.pos_sigma_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 5. Vector Readout for denoising: v [N, F, 3] -> [N, 3]
        self.vector_readout = nn.Linear(hidden_dim, 1, bias=False)

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
        v = torch.zeros(N, F, 3, device=s.device, dtype=s.dtype)  # [N, F, 3]

        # --- 3. Per-node conditioning ---
        prec_expanded = global_prec[batch]                   # [N, 12]
        if self.num_cheap_dft_inputs > 0:
            node_cheap_dft = data.cheap_dft_scalars          # [N, C]
            cond = torch.cat([node_cheap_dft, prec_expanded], dim=1)
        else:
            cond = prec_expanded                             # [N, 12]

        # --- 4. PaiNN Message Passing ---
        for layer in self.layers:
            s, v = layer(s, v, edge_index, pos, cond)

        # --- 5. Vector output (for denoising) ---
        # v: [N, F, 3] -> [N, 3, F] -> linear -> [N, 3, 1] -> [N, 3]
        vector_output = self.vector_readout(v.transpose(1, 2)).squeeze(-1)
        self._vector_output = vector_output

        # --- 6. Position prediction ---
        delta_positions_mean = vector_output                  # [N, 3]
        delta_positions_logvar = self.pos_sigma_head(s)       # [N, 1]

        # --- 7. Graph Pooling & Readout ---
        h_graph_sum = global_add_pool(s, batch)               # [B, F]
        h_graph_mean = global_mean_pool(s, batch)             # [B, F]

        prec_readout = self.settings_embed(global_prec)       # [B, F]

        # A. Energy (extensive -> sum pool)
        readout_parts = [h_graph_sum, prec_readout]
        if self.num_cheap_dft_inputs > 0:
            graph_dft = global_mean_pool(node_cheap_dft, batch)  # [B, C]
            readout_parts.append(graph_dft)
        in_energy = torch.cat(readout_parts, dim=1)
        out_energy = self.energy_head(in_energy)
        delta_energy_mean = out_energy[:, 0]
        delta_energy_logvar = out_energy[:, 1]

        # B. Gap (intensive -> mean pool)
        gap_parts = [h_graph_mean, prec_readout]
        if self.num_cheap_dft_inputs > 0:
            gap_parts.append(graph_dft)
        in_gap = torch.cat(gap_parts, dim=1)
        out_gap = self.bandgap_head(in_gap)
        delta_gap_mean = out_gap[:, 0]
        delta_gap_logvar = out_gap[:, 1]

        # C. Geometry
        geo_parts = [h_graph_mean, prec_readout]
        if self.num_cheap_dft_inputs > 0:
            geo_parts.append(graph_dft)
        if self.num_geo_inputs > 0:
            cheap_geo = data.cheap_geometry_scalars           # [B, 7]
            geo_parts.append(cheap_geo)
        in_geo = torch.cat(geo_parts, dim=1)
        out_geo = self.geometry_head(in_geo)
        delta_geo_mean, delta_geo_logvar = out_geo.chunk(2, dim=1)

        # --- 8. Return (same format as DeltaGNN) ---
        return (delta_positions_mean, delta_positions_logvar), \
               (delta_energy_mean, delta_energy_logvar), \
               (delta_gap_mean, delta_gap_logvar), \
               (delta_geo_mean, delta_geo_logvar)
