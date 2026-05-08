import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, softmax
from torch_geometric.utils import scatter

from .painn_layer import FiLMGenerator


class EGNNAttentionLayer(MessagePassing):
    """
    Equivariant Graph Neural Network Layer with Softmax Attention.

    Extends the base EGNNLayer by adding learned attention weights that
    modulate both feature messages and coordinate updates per-edge.
    Optionally supports FiLM conditioning (gamma * h + beta) before the
    node update MLP, matching the base EGNNLayer's FiLM mechanism.

    Flow:
      m_ij       = phi_m(h_i, h_j, ||x_i - x_j||^2)   # message (invariant)
      score_ij   = phi_att(m_ij)                         # attention logit
      alpha_ij   = softmax_j(score_ij)                   # normalized per target node
      x_update   = phi_x(m_ij) * (x_i - x_j)            # coord update (equivariant)

      agg_m_i    = sum_j(alpha_ij * m_ij)                # attention-weighted messages
      agg_x_i    = sum_j(alpha_ij * x_update)            # attention-weighted coord updates

      h_mod      = h_i * gamma + beta  (if FiLM)         # FiLM modulation
      h_i'       = phi_h(h_mod, agg_m_i) + h_i           # node update + residual
      x_i'       = x_i + agg_x_i                         # coord update

    E(3)-equivariance: alpha_ij is a scalar computed from invariant quantities
    (h features, squared distance). Multiplying an equivariant vector by an
    invariant scalar preserves equivariance.
    """
    def __init__(self, in_features_h, hidden_features, out_features_h,
                 use_film=False, cond_dim=0, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.use_film = use_film
        if use_film:
            self.film = FiLMGenerator(cond_dim, in_features_h)

        # phi_m: Message network (h_i, h_j, ||x_i - x_j||^2) -> message vector
        self.phi_m = nn.Sequential(
            nn.Linear(2 * in_features_h + 1, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features)
        )

        # phi_att: Attention score network (message -> scalar logit)
        self.phi_att = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 1)
        )

        # phi_x: Position update magnitude (message -> scalar)
        self.phi_x = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 1)
        )

        # phi_h: Node update network (h_i, aggregated_message) -> h_i_new
        self.phi_h = nn.Sequential(
            nn.Linear(in_features_h + hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features_h)
        )

        self.norm = nn.LayerNorm(out_features_h)
        self.residuals = (in_features_h == out_features_h)

        # Near-zero init for position updates to prevent early explosion
        nn.init.uniform_(self.phi_x[-1].weight, a=-1e-3, b=1e-3)
        nn.init.uniform_(self.phi_x[-1].bias, a=-1e-3, b=1e-3)

    def forward(self, h, x, edge_index, cond=None):
        edge_index, _ = remove_self_loops(edge_index)

        # Store conditioning for use in update()
        self._cond = cond

        h_out, x_out = self.propagate(edge_index, h=h, x=x)
        self._cond = None  # Clear to prevent stale state

        if self.residuals:
            h_out = h_out + h
        h_out = self.norm(h_out)

        return h_out, x_out

    def message(self, h_i, h_j, x_i, x_j):
        # Squared distance (invariant scalar)
        x_diff = x_i - x_j
        dist_sq = (x_diff ** 2).sum(dim=1, keepdim=True)

        # Compute message
        phi_m_in = torch.cat([h_i, h_j, dist_sq], dim=1)
        m_ij = self.phi_m(phi_m_in)

        # Attention logit (unnormalized — softmax applied in aggregate)
        score_ij = self.phi_att(m_ij)  # [E, 1]

        # Position update (before attention weighting)
        magnitude = self.phi_x(m_ij)   # [E, 1]
        x_update = magnitude * x_diff  # [E, 3]

        return m_ij, x_update, score_ij

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        m_ij, x_update, score_ij = inputs

        # Softmax attention: normalize scores per target node
        att_weights = softmax(score_ij, index)  # [E, 1]

        # Weight both feature messages and coordinate updates
        weighted_m = att_weights * m_ij          # [E, hidden]
        weighted_x = att_weights * x_update      # [E, 3]

        # Scatter-sum (attention-weighted aggregation)
        m_i = scatter(weighted_m, index, dim=self.node_dim,
                      dim_size=dim_size, reduce='sum')
        x_update_agg = scatter(weighted_x, index, dim=self.node_dim,
                               dim_size=dim_size, reduce='sum')

        return m_i, x_update_agg

    def update(self, aggr_out, h, x):
        m_i, x_update_term = aggr_out

        # FiLM modulation: h_mod = h * gamma + beta
        if self.use_film and self._cond is not None:
            gamma, beta = self.film(self._cond)
            h_mod = h * gamma + beta
        else:
            h_mod = h

        # Scalar feature update (invariant)
        phi_h_in = torch.cat([h_mod, m_i], dim=1)
        h_out = self.phi_h(phi_h_in)

        # Position update (equivariant)
        x_out = x + x_update_term

        return h_out, x_out
