import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax
from .egnn_attention_layer import EGNNAttentionLayer


class DeltaAttentionGNN(nn.Module):
    """
    Delta EGNN with attention in both message passing and graph pooling.

    Compared to DeltaGNN:
    - Message passing uses EGNNAttentionLayer (softmax attention over neighbors)
    - Graph pooling uses learned attention weights instead of uniform mean/sum
    - Everything else (embeddings, conditioning, readout heads) is identical
    """
    def __init__(self,
                 num_layers,
                 hidden_features,
                 num_cheap_dft_inputs,
                 num_precision_settings=13,
                 num_geo_inputs=7,
                 max_z=100,
                 use_film=False):
        super().__init__()

        self.num_cheap_dft_inputs = num_cheap_dft_inputs
        self.num_geo_inputs = num_geo_inputs
        self.use_film = use_film

        # --- Dimensions (same as DeltaGNN) ---
        self.z_dim = hidden_features
        self.cond_dim = hidden_features
        self.egnn_dim = self.z_dim + self.cond_dim  # = 2 * hidden_features

        # FiLM conditioning dimension (raw cheap_dft + precision_settings per node)
        film_cond_dim = num_cheap_dft_inputs + num_precision_settings

        # --- Embeddings (same as DeltaGNN) ---
        self.z_embedding = nn.Embedding(max_z, self.z_dim)

        conditioning_input_size = num_cheap_dft_inputs + num_precision_settings
        self.conditioning_proj = nn.Sequential(
            nn.Linear(conditioning_input_size, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim)
        )

        # --- Message Passing (attention layers instead of base EGNN) ---
        self.layers = nn.ModuleList([
            EGNNAttentionLayer(self.egnn_dim, self.egnn_dim, self.egnn_dim,
                               use_film=use_film, cond_dim=film_cond_dim)
            for _ in range(num_layers)
        ])

        # --- Readout Components ---

        self.settings_embed = nn.Linear(num_precision_settings, self.cond_dim)

        base_readout_size = self.egnn_dim + self.cond_dim + num_cheap_dft_inputs

        # Attention-based graph pooling gates
        # Separate gates for energy (extensive) and gap/geo (intensive)
        self.energy_pool_gate = nn.Sequential(
            nn.Linear(self.egnn_dim, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 1)
        )
        self.intensive_pool_gate = nn.Sequential(
            nn.Linear(self.egnn_dim, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 1)
        )

        # 1. Delta Energy Head (Outputs 2: Mean, LogVar)
        self.energy_head = nn.Sequential(
            nn.Linear(base_readout_size, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 2)
        )

        # 2. Delta Bandgap Head (Outputs 2: Mean, LogVar)
        self.bandgap_head = nn.Sequential(
            nn.Linear(base_readout_size, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 2)
        )

        # 3. Geometry Head (Outputs 14: 7 Means + 7 LogVars)
        geo_input_size = base_readout_size + num_geo_inputs
        self.geometry_head = nn.Sequential(
            nn.Linear(geo_input_size, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 7 * 2)  # Always 7 geo targets (vol + 6 lattice)
        )

        # 4. Position Uncertainty Head
        self.pos_sigma_head = nn.Sequential(
            nn.Linear(self.egnn_dim, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 1)
        )

    def _attention_pool(self, gate, h, batch):
        """Attention-weighted graph pooling.

        Computes per-node softmax attention weights and returns
        the weighted sum over each graph.
        """
        scores = gate(h)                                 # [N, 1]
        weights = pyg_softmax(scores, batch)             # [N, 1]
        return global_add_pool(weights * h, batch)       # [B, egnn_dim]

    def forward(self, data):
        """
        Returns tuples: (mean, log_var) for each target.
        Same output format as DeltaGNN for drop-in compatibility.
        """

        # --- 1. Unpack Inputs ---

        pos = data.x                                      # [N, 3]
        batch = data.batch                                # [N]
        edge_index = data.edge_index                      # [2, E]
        z = data.z.long().squeeze()                       # [N]

        global_precision_settings = data.precision_settings  # [B, 12]

        pos_init = pos.clone()

        # --- 2. Feature Embedding & Conditioning ---

        h_z = self.z_embedding(z)

        precision_settings_expanded = global_precision_settings[batch]

        if self.num_cheap_dft_inputs > 0:
            node_cheap_dft_inputs = data.cheap_dft_scalars  # [N, C]
            raw_cond = torch.cat([node_cheap_dft_inputs, precision_settings_expanded], dim=1)
        else:
            raw_cond = precision_settings_expanded

        h_cond = self.conditioning_proj(raw_cond)

        h_egnn = torch.cat([h_z, h_cond], dim=1)

        # FiLM conditioning: pass raw per-node features to each layer
        film_cond = raw_cond if self.use_film else None

        # --- 3. Attention EGNN Message Passing ---
        for layer in self.layers:
            h_egnn, pos = layer(h_egnn, pos, edge_index, cond=film_cond)

        # --- 4. Prediction: Positions ---
        delta_positions_mean = pos - pos_init
        delta_positions_logvar = self.pos_sigma_head(h_egnn)

        # --- 5. Attention Graph Pooling & Readout Preparation ---

        h_graph_energy = self._attention_pool(
            self.energy_pool_gate, h_egnn, batch)         # [B, egnn_dim]
        h_graph_intensive = self._attention_pool(
            self.intensive_pool_gate, h_egnn, batch)      # [B, egnn_dim]

        precision_settings_readout = self.settings_embed(global_precision_settings)

        # --- 6. Prediction Heads ---

        # A. Energy Delta
        readout_parts = [h_graph_energy, precision_settings_readout]
        if self.num_cheap_dft_inputs > 0:
            graph_dft_scalars = global_mean_pool(node_cheap_dft_inputs, batch)
            readout_parts.append(graph_dft_scalars)
        in_energy = torch.cat(readout_parts, dim=1)
        out_energy = self.energy_head(in_energy)
        delta_energy_mean = out_energy[:, 0]
        delta_energy_logvar = out_energy[:, 1]

        # B. Gap Delta
        gap_parts = [h_graph_intensive, precision_settings_readout]
        if self.num_cheap_dft_inputs > 0:
            gap_parts.append(graph_dft_scalars)
        in_gap = torch.cat(gap_parts, dim=1)
        out_gap = self.bandgap_head(in_gap)
        delta_gap_mean = out_gap[:, 0]
        delta_gap_logvar = out_gap[:, 1]

        # C. Geometry Delta
        geo_parts = [h_graph_intensive, precision_settings_readout]
        if self.num_cheap_dft_inputs > 0:
            geo_parts.append(graph_dft_scalars)
        if self.num_geo_inputs > 0:
            cheap_geo = data.cheap_geometry_scalars         # [B, 7]
            geo_parts.append(cheap_geo)
        in_geo = torch.cat(geo_parts, dim=1)
        out_geo = self.geometry_head(in_geo)
        delta_geo_mean, delta_geo_logvar = out_geo.chunk(2, dim=1)

        # --- 7. Return Tuples ---
        return (delta_positions_mean, delta_positions_logvar), \
               (delta_energy_mean, delta_energy_logvar), \
               (delta_gap_mean, delta_gap_logvar), \
               (delta_geo_mean, delta_geo_logvar)
