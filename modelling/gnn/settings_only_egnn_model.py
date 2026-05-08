import torch
from torch import nn
from torch_geometric.nn import global_mean_pool, global_add_pool
from .egnn_layer import EGNNLayer

class SettingsOnlyGNN(nn.Module):
    """
    Ablation variant of DeltaGNN that uses only atomic structure and DFT
    precision settings (binary_precision + k_point_density) as inputs.
    Does NOT use cheap DFT results (energies, eigenvalues, etc.) or
    cheap geometry scalars (volume, lattice params).
    """
    def __init__(self,
                 num_layers,
                 hidden_features,
                 num_precision_settings=13,
                 num_geo_inputs=7,
                 max_z=100):
        super().__init__()

        # --- Dimensions ---
        self.z_dim = hidden_features
        self.cond_dim = hidden_features
        self.egnn_dim = self.z_dim + self.cond_dim # = 2 * hidden_features

        # --- Embeddings ---
        self.z_embedding = nn.Embedding(max_z, self.z_dim)

        # Conditioning Projector: only precision settings, no DFT scalars
        self.conditioning_proj = nn.Sequential(
            nn.Linear(num_precision_settings, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim)
        )

        # --- Message Passing ---
        self.layers = nn.ModuleList([
            EGNNLayer(self.egnn_dim, self.egnn_dim, self.egnn_dim)
            for _ in range(num_layers)
        ])

        # --- Readout Components ---

        self.settings_embed = nn.Linear(num_precision_settings, self.cond_dim)

        # No cheap DFT scalars in readout
        base_readout_size = self.egnn_dim + self.cond_dim

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
        # No cheap_geo concatenated, so same as base_readout_size
        self.geometry_head = nn.Sequential(
            nn.Linear(base_readout_size, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 7 * 2)  # Always 7 geo targets (vol + 6 lattice)
        )

        # 4. Position Uncertainty Head
        self.pos_sigma_head = nn.Sequential(
            nn.Linear(self.egnn_dim, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 1) # Predicts 1 log_var per atom
        )


    def forward(self, data, return_node_features=False):
        """
        Returns tuples: (mean, log_var) for each target.
        Same output format as DeltaGNN for drop-in compatibility.

        If return_node_features=True, returns h_egnn [N, egnn_dim] instead
        (per-node features after message passing, before pooling).
        """

        # --- 1. Unpack Inputs (no cheap DFT scalars needed) ---

        pos = data.x                     # [N, 3]
        batch = data.batch               # [N]
        edge_index = data.edge_index     # [2, E]
        z = data.z.long().squeeze()      # [N]

        global_precision_settings = data.precision_settings # [B, 12]

        pos_init = pos.clone()

        # --- 2. Feature Embedding & Conditioning ---

        h_z = self.z_embedding(z)  # [N, hidden_dim]

        precision_settings_expanded = global_precision_settings[batch]  # [N, 12]

        h_cond = self.conditioning_proj(precision_settings_expanded)

        h_egnn = torch.cat([h_z, h_cond], dim=1)

        # --- 3. EGNN Message Passing ---
        for layer in self.layers:
            h_egnn, pos = layer(h_egnn, pos, edge_index)

        # --- Early return for embedding extraction ---
        if return_node_features:
            return h_egnn  # [N, egnn_dim]

        # --- 4. Prediction: Positions ---
        delta_positions_mean = pos - pos_init     # [N, 3]
        delta_positions_logvar = self.pos_sigma_head(h_egnn) # [N, 1]

        # --- 5. Graph Pooling & Readout Preparation ---

        h_graph_sum = global_add_pool(h_egnn, batch)   # [B, egnn_dim]
        h_graph_mean = global_mean_pool(h_egnn, batch) # [B, egnn_dim]

        precision_settings_readout = self.settings_embed(global_precision_settings) # [B, cond_dim]

        # --- 6. Prediction Heads (Scalars & Geo) ---

        # A. Energy Delta
        in_energy = torch.cat([
            h_graph_sum, precision_settings_readout], dim=1)
        out_energy = self.energy_head(in_energy) # [B, 2]
        delta_energy_mean = out_energy[:, 0]
        delta_energy_logvar = out_energy[:, 1]

        # B. Gap Delta
        in_gap = torch.cat([
            h_graph_mean, precision_settings_readout], dim=1)
        out_gap = self.bandgap_head(in_gap) # [B, 2]
        delta_gap_mean = out_gap[:, 0]
        delta_gap_logvar = out_gap[:, 1]

        # C. Geometry Delta (no cheap_geo input)
        in_geo = torch.cat([h_graph_mean, precision_settings_readout], dim=1)
        out_geo = self.geometry_head(in_geo) # [B, 14]

        delta_geo_mean, delta_geo_logvar = out_geo.chunk(2, dim=1)

        # --- 7. Return Tuples ---
        return (delta_positions_mean, delta_positions_logvar), \
               (delta_energy_mean, delta_energy_logvar), \
               (delta_gap_mean, delta_gap_logvar), \
               (delta_geo_mean, delta_geo_logvar)
