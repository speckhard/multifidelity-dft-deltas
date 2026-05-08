"""
PaiNN building blocks with per-edge attention in message aggregation.

Extends painn_layer.py by adding softmax attention over neighbor messages.
Attention scores are computed from invariant quantities (RBF distances +
scalar features), preserving E(3) equivariance when weighting vector messages.

The PaiNNUpdate block is reused unchanged from painn_layer.py.
"""

import torch
from torch import nn
from torch_geometric.utils import softmax as pyg_softmax
from torch_geometric.utils import scatter

from .painn_layer import GaussianRBF, CosineCutoff, PaiNNUpdate, FiLMGenerator


class PaiNNAttentionMessage(nn.Module):
    """PaiNN message passing with per-edge softmax attention.

    Like PaiNNMessage, but computes attention logits from RBF features
    and source/target scalar features, then weights messages before
    aggregation.

    Scalar: ds_i = sum_j alpha_ij * filter_s(rbf_ij) * s_j
    Vector: dv_i = sum_j alpha_ij * [filter_v1(rbf_ij) * v_j
                                    + filter_v2(rbf_ij) * s_j * unit_dir_ij]

    where alpha_ij = softmax_j(phi_att(rbf_ij, s_i, s_j))
    """

    def __init__(self, hidden_dim: int, num_rbf: int, cutoff: float):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Same filter network as PaiNNMessage
        self.filter_net = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim)
        )
        self.rbf = GaussianRBF(num_rbf, cutoff)
        self.cutoff_fn = CosineCutoff(cutoff)

        # Attention score network: (rbf, s_target, s_source) -> scalar logit
        # Input: num_rbf + 2*hidden_dim (target scalar + source scalar)
        self.attention_net = nn.Sequential(
            nn.Linear(num_rbf + 2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, s, v, edge_index, pos):
        # s: [N, F], v: [N, F, 3], pos: [N, 3]
        row, col = edge_index  # row=target, col=source

        edge_vec = pos[row] - pos[col]                     # [E, 3]
        dist = edge_vec.norm(dim=-1)                        # [E]
        unit_dir = edge_vec / (dist.unsqueeze(-1) + 1e-8)   # [E, 3]

        rbf_out = self.rbf(dist)                             # [E, num_rbf]
        cutoff_w = self.cutoff_fn(dist)                      # [E]

        filters = self.filter_net(rbf_out) * cutoff_w.unsqueeze(-1)  # [E, 3F]
        filter_s, filter_v1, filter_v2 = filters.split(self.hidden_dim, dim=-1)

        # Gather source features
        s_j = s[col]   # [E, F]
        s_i = s[row]   # [E, F]
        v_j = v[col]   # [E, F, 3]

        # --- Attention ---
        att_input = torch.cat([rbf_out, s_i, s_j], dim=-1)  # [E, num_rbf + 2F]
        att_logit = self.attention_net(att_input)              # [E, 1]
        att_weights = pyg_softmax(att_logit, row)              # [E, 1] softmax per target node

        # Scalar messages (attention-weighted)
        s_msg = att_weights * filter_s * s_j  # [E, F]

        # Vector messages (attention-weighted)
        v_msg_raw = (filter_v1.unsqueeze(-1) * v_j
                     + (filter_v2 * s_j).unsqueeze(-1) * unit_dir.unsqueeze(1))
        v_msg = att_weights.unsqueeze(-1) * v_msg_raw  # [E, F, 3]

        # Aggregate to target nodes
        ds = scatter(s_msg, row, dim=0, dim_size=s.size(0), reduce='sum')
        dv = scatter(v_msg, row, dim=0, dim_size=v.size(0), reduce='sum')

        return ds, dv


class PaiNNAttentionInteraction(nn.Module):
    """PaiNN interaction layer with attention messages + FiLM update + residual."""

    def __init__(self, hidden_dim: int, num_rbf: int, cutoff: float, cond_dim: int,
                 use_film: bool = True):
        super().__init__()
        self.message_block = PaiNNAttentionMessage(hidden_dim, num_rbf, cutoff)
        self.update_block = PaiNNUpdate(hidden_dim, cond_dim, use_film=use_film)
        self.scalar_norm = nn.LayerNorm(hidden_dim)

    def forward(self, s, v, edge_index, pos, cond):
        # Message block (with residual)
        ds_msg, dv_msg = self.message_block(s, v, edge_index, pos)
        s = s + ds_msg
        v = v + dv_msg

        # Update block (with residual)
        ds_upd, dv_upd = self.update_block(s, v, cond)
        s = s + ds_upd
        v = v + dv_upd

        s = self.scalar_norm(s)

        return s, v
