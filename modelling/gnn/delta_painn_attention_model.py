"""
DeltaPaiNNAttention: PaiNN with per-edge attention + FiLM + attention pooling.

Combines PaiNN's separate scalar/vector tracks and FiLM conditioning with:
1. Per-edge softmax attention in message aggregation (from attention EGNN)
2. Learned attention-weighted graph pooling (from DeltaAttentionGNN)

Drop-in replacement for DeltaPaiNN / DeltaGNN with identical forward(data)
signature and return format.
"""

import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax
from .painn_attention_layer import PaiNNAttentionInteraction


class DeltaPaiNNAttention(nn.Module):

    def __init__(self,
                 num_layers: int,
                 hidden_dim: int,
                 num_cheap_dft_inputs: int = 12,
                 num_precision_settings: int = 12,
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

        # --- PaiNN Attention Interaction Layers ---
        self.layers = nn.ModuleList([
            PaiNNAttentionInteraction(hidden_dim, num_rbf, cutoff, cond_dim,
                                      use_film=use_film)
            for _ in range(num_layers)
        ])

        # --- Readout ---
        self.settings_embed = nn.Linear(num_precision_settings, hidden_dim)

        base_readout_size = hidden_dim + hidden_dim + num_cheap_dft_inputs

        # Attention-based graph pooling gates (from DeltaAttentionGNN)
        self.energy_pool_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.intensive_pool_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

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
            nn.Linear(hidden_dim, 7 * 2)
        )

        # 4. Position Uncertainty Head (per-atom logvar from scalars)
        self.pos_sigma_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 5. Vector Readout for position prediction: v [N, F, 3] -> [N, 3]
        self.vector_readout = nn.Linear(hidden_dim, 1, bias=False)

    def _attention_pool(self, gate, h, batch):
        """Attention-weighted graph pooling (softmax over nodes per graph)."""
        scores = gate(h)                                 # [N, 1]
        weights = pyg_softmax(scores, batch)             # [N, 1]
        return global_add_pool(weights * h, batch)       # [B, hidden_dim]

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

        # --- 3. Per-node conditioning ---
        prec_expanded = global_prec[batch]                   # [N, 12]
        if self.num_cheap_dft_inputs > 0:
            node_cheap_dft = data.cheap_dft_scalars          # [N, C]
            cond = torch.cat([node_cheap_dft, prec_expanded], dim=1)
        else:
            cond = prec_expanded

        # --- 4. PaiNN Attention Message Passing ---
        for layer in self.layers:
            s, v = layer(s, v, edge_index, pos, cond)

        # --- 5. Vector output (position prediction) ---
        vector_output = self.vector_readout(v.transpose(1, 2)).squeeze(-1)
        self._vector_output = vector_output

        # --- 6. Position prediction ---
        delta_positions_mean = vector_output                  # [N, 3]
        delta_positions_logvar = self.pos_sigma_head(s)       # [N, 1]

        # --- 7. Attention Graph Pooling & Readout ---
        h_graph_energy = self._attention_pool(
            self.energy_pool_gate, s, batch)                  # [B, F]
        h_graph_intensive = self._attention_pool(
            self.intensive_pool_gate, s, batch)               # [B, F]

        prec_readout = self.settings_embed(global_prec)       # [B, F]

        # A. Energy (extensive -> attention-weighted pool)
        readout_parts = [h_graph_energy, prec_readout]
        if self.num_cheap_dft_inputs > 0:
            graph_dft = global_mean_pool(node_cheap_dft, batch)
            readout_parts.append(graph_dft)
        in_energy = torch.cat(readout_parts, dim=1)
        out_energy = self.energy_head(in_energy)
        delta_energy_mean = out_energy[:, 0]
        delta_energy_logvar = out_energy[:, 1]

        # B. Gap (intensive -> attention-weighted pool)
        gap_parts = [h_graph_intensive, prec_readout]
        if self.num_cheap_dft_inputs > 0:
            gap_parts.append(graph_dft)
        in_gap = torch.cat(gap_parts, dim=1)
        out_gap = self.bandgap_head(in_gap)
        delta_gap_mean = out_gap[:, 0]
        delta_gap_logvar = out_gap[:, 1]

        # C. Geometry
        geo_parts = [h_graph_intensive, prec_readout]
        if self.num_cheap_dft_inputs > 0:
            geo_parts.append(graph_dft)
        if self.num_geo_inputs > 0:
            cheap_geo = data.cheap_geometry_scalars
            geo_parts.append(cheap_geo)
        in_geo = torch.cat(geo_parts, dim=1)
        out_geo = self.geometry_head(in_geo)
        delta_geo_mean, delta_geo_logvar = out_geo.chunk(2, dim=1)

        # --- 8. Return (same format as DeltaGNN/DeltaPaiNN) ---
        return (delta_positions_mean, delta_positions_logvar), \
               (delta_energy_mean, delta_energy_logvar), \
               (delta_gap_mean, delta_gap_logvar), \
               (delta_geo_mean, delta_geo_logvar)
