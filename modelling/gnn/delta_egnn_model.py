import torch
from torch import nn
from torch_geometric.nn import global_mean_pool, global_add_pool
from .egnn_layer import EGNNLayer

class DeltaGNN(nn.Module):
    def __init__(self,
                 num_layers,
                 hidden_features,
                 num_cheap_dft_inputs,
                 num_precision_settings=13,
                 num_geo_inputs=7,
                 max_z=100,
                 use_film=False):
        super().__init__()
        self.use_film = use_film
        self.num_cheap_dft_inputs = num_cheap_dft_inputs
        self.num_geo_inputs = num_geo_inputs

        # --- Dimensions ---
        self.z_dim = hidden_features
        self.cond_dim = hidden_features
        self.egnn_dim = self.z_dim + self.cond_dim # = 2 * hidden_features

        # --- Embeddings ---
        self.z_embedding = nn.Embedding(max_z, self.z_dim)

        # Conditioning Projector
        conditioning_input_size = num_cheap_dft_inputs + num_precision_settings

        self.conditioning_proj = nn.Sequential(
            nn.Linear(conditioning_input_size, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim)
        )

        # --- Message Passing ---
        self.layers = nn.ModuleList([
            EGNNLayer(self.egnn_dim, self.egnn_dim, self.egnn_dim,
                      use_film=use_film, cond_dim=conditioning_input_size)
            for _ in range(num_layers)
        ])
        
        # --- Readout Components ---
        
        self.settings_embed = nn.Linear(num_precision_settings, self.cond_dim)
        
        base_readout_size = self.egnn_dim + self.cond_dim + num_cheap_dft_inputs
        
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
        # FIX: Input dim must be self.egnn_dim (state after message passing)
        self.pos_sigma_head = nn.Sequential(
            nn.Linear(self.egnn_dim, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 1) # Predicts 1 log_var per atom
        )


    def forward(self, data):
        """
        Returns tuples: (mean, log_var) for each target.
        """
        
        # --- 1. Unpack and Standardize Inputs ---

        pos = data.x                     # [N, 3]
        batch = data.batch               # [N]
        edge_index = data.edge_index     # [2, E]
        z = data.z.long().squeeze()      # [N]

        global_precision_settings = data.precision_settings # [B, 12]

        pos_init = pos.clone()

        # --- 2. Feature Embedding & Conditioning ---

        h_z = self.z_embedding(z)  # [N, hidden_dim]

        precision_settings_expanded = global_precision_settings[batch]  # [N, 12]

        if self.num_cheap_dft_inputs > 0:
            node_cheap_dft_inputs = data.cheap_dft_scalars  # [N, C]
            raw_cond = torch.cat([node_cheap_dft_inputs, precision_settings_expanded], dim=1)
        else:
            raw_cond = precision_settings_expanded  # [N, 12]

        h_cond = self.conditioning_proj(raw_cond)

        h_egnn = torch.cat([h_z, h_cond], dim=1)

        # --- 3. EGNN Message Passing ---
        film_cond = raw_cond if self.use_film else None
        for layer in self.layers:
            h_egnn, pos = layer(h_egnn, pos, edge_index, cond=film_cond)

        # --- 4. Prediction: Positions ---
        delta_positions_mean = pos - pos_init     # [N, 3]
        delta_positions_logvar = self.pos_sigma_head(h_egnn) # [N, 1]

        # --- 5. Graph Pooling & Readout Preparation ---

        h_graph_sum = global_add_pool(h_egnn, batch)   # [B, egnn_dim]
        h_graph_mean = global_mean_pool(h_egnn, batch) # [B, egnn_dim]

        precision_settings_readout = self.settings_embed(global_precision_settings) # [B, cond_dim]

        if self.num_cheap_dft_inputs > 0:
            graph_dft_scalars = global_mean_pool(node_cheap_dft_inputs, batch)  # [B, C]

        # --- 6. Prediction Heads (Scalars & Geo) ---

        # A. Energy Delta
        readout_parts = [h_graph_sum, precision_settings_readout]
        if self.num_cheap_dft_inputs > 0:
            readout_parts.append(graph_dft_scalars)
        in_energy = torch.cat(readout_parts, dim=1)
        out_energy = self.energy_head(in_energy) # [B, 2]
        delta_energy_mean = out_energy[:, 0]
        delta_energy_logvar = out_energy[:, 1]

        # B. Gap Delta
        readout_parts = [h_graph_mean, precision_settings_readout]
        if self.num_cheap_dft_inputs > 0:
            readout_parts.append(graph_dft_scalars)
        in_gap = torch.cat(readout_parts, dim=1)
        out_gap = self.bandgap_head(in_gap) # [B, 2]
        delta_gap_mean = out_gap[:, 0]
        delta_gap_logvar = out_gap[:, 1]

        # C. Geometry Delta
        geo_parts = [h_graph_mean, precision_settings_readout]
        if self.num_cheap_dft_inputs > 0:
            geo_parts.append(graph_dft_scalars)
        if self.num_geo_inputs > 0:
            cheap_geo = data.cheap_geometry_scalars  # [B, 7]
            geo_parts.append(cheap_geo)
        in_geo = torch.cat(geo_parts, dim=1)
        out_geo = self.geometry_head(in_geo) # [B, 14]
        
        # Split into Mean [B, 7] and LogVar [B, 7]
        # We assume first 7 outputs are mean, last 7 are logvar
        delta_geo_mean, delta_geo_logvar = out_geo.chunk(2, dim=1)

        # --- 7. Return Tuples ---
        return (delta_positions_mean, delta_positions_logvar), \
               (delta_energy_mean, delta_energy_logvar), \
               (delta_gap_mean, delta_gap_logvar), \
               (delta_geo_mean, delta_geo_logvar)
