import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import scatter # Needed for manual aggregation

from .painn_layer import FiLMGenerator


class EGNNLayer(MessagePassing):
    """
    Equivariant Graph Neural Network Layer (EGNN).
    
    Guarantees E(3)-equivariance for the coordinate vectors (x) and
    invariance for the scalar features (h).
    """
    def __init__(self, in_features_h, hidden_features, out_features_h,
                 use_film=False, cond_dim=0, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.use_film = use_film
        if use_film:
            self.film = FiLMGenerator(cond_dim, in_features_h)

        # message (mij for edge i,j)
        # Phi_m: Message network (h_i, h_j, ||x_i - x_j||^2) -> message vector
        # Input size: 2 * in_features_h + 1 (for squared distance)
        self.phi_m = nn.Sequential(
            nn.Linear(2 * in_features_h + 1, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features) # Output: message vector m_ij
        )

        # Determines how much the position of the node should move away from
        # i,j vector. How much atom i should move away from atom j.
        # Phi_x: Network for position update magnitude (message vector -> scalar magnitude)
        # Output size is 1, representing the scalar magnitude of displacement along the edge vector.
        # NOTE, we removed the sigmoid at the end of this to ensure phi_x can be negative.
        self.phi_x = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 1))

        # Node update equation.
        # Phi_h: Update network for scalar features (h_i, sum(m_ij)) -> h_i_new
        # Input size: in_features_h + hidden_features (for aggregated message)
        self.phi_h = nn.Sequential(
            nn.Linear(in_features_h + hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features_h)
        )
        
        # Final normalization
        self.norm = nn.LayerNorm(out_features_h)
        # Check for residual connection validity
        self.residuals = (in_features_h == out_features_h)
        # Initialize the last linear layer of phi_x (position update) to near zero.
        # This makes sure that during training we don't explode in initial predictions.
        nn.init.uniform_(self.phi_x[-1].weight, a=-1e-3, b=1e-3)
        nn.init.uniform_(self.phi_x[-1].bias, a=-1e-3, b=1e-3)
        # Note: phi_x has no Sigmoid — magnitude can be negative


    def forward(self, h, x, edge_index, cond=None):
        # 1. Remove self-loops for message passing
        edge_index, _ = remove_self_loops(edge_index)

        # Store conditioning for use in update()
        self._cond = cond

        # Propagate: message(), aggregate(), update()
        h_out, x_out = self.propagate(edge_index, h=h, x=x)
        self._cond = None  # Clear to prevent stale state on next call

        # Residual + LayerNorm
        if self.residuals:
            h_out = h_out + h
        h_out = self.norm(h_out)

        return h_out, x_out


    def message(self, h_i, h_j, x_i, x_j):
        # Compute vector e_ij and squared distance ||x_i - x_j||^2
        # The vector difference, rotates if the graph rotates, equivariant
        # R phi (v) = phi(R v), translation ignored but true also.
        x_diff = x_i - x_j
        # The distance is invariant to rotations and is a scalar.
        dist_sq = (x_diff ** 2).sum(dim=1, keepdim=True)

        # Concatenate scalar features and squared distance
        phi_m_in = torch.cat([h_i, h_j, dist_sq], dim=1)
        
        # Calculate message vector m_ij
        m_ij = self.phi_m(phi_m_in)
        
        # Calculate scalar magnitude for position update (alpha_ij)
        alpha_ij = self.phi_x(m_ij)
        
        # Return message vector (m_ij) and the term needed for position update: alpha_ij * e_ij
        return m_ij, alpha_ij * x_diff


    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        # inputs is the tuple returned by message()
        m_ij, x_update_msg = inputs
        
        # Aggregate feature messages (sum)
        m_i = scatter(m_ij, index, dim=self.node_dim, dim_size=dim_size, reduce='sum')
        
        # Aggregate coordinate messages (sum)
        x_update_agg = scatter(x_update_msg, index, dim=self.node_dim, dim_size=dim_size, reduce='sum')
        
        return m_i, x_update_agg


    def update(self, aggr_out, h, x):
        # aggr_out is a tuple: (aggregated messages, aggregated position update terms)
        m_i, x_update_term = aggr_out

        # FiLM modulation: h_mod = h * gamma + beta
        if self.use_film and self._cond is not None:
            gamma, beta = self.film(self._cond)
            h_mod = h * gamma + beta
        else:
            h_mod = h

        # 1. Scalar Feature Update (Invariant)
        phi_h_in = torch.cat([h_mod, m_i], dim=1)
        h_out = self.phi_h(phi_h_in)

        # 2. Position Update (Equivariant)
        x_out = x + x_update_term

        return h_out, x_out
