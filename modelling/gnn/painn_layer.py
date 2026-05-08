"""
PaiNN (Polarizable Atom Interaction Neural Network) building blocks
with FiLM (Feature-wise Linear Modulation) conditioning.

Reference: Schütt et al., "Equivariant message passing for the prediction
of tensorial properties and molecular spectra", ICML 2021.

Key: PaiNN maintains separate scalar (s) and vector (v) feature tracks.
Scalars are invariant, vectors are equivariant under E(3) rotations.
"""

import torch
from torch import nn
from torch_geometric.utils import scatter


class GaussianRBF(nn.Module):
    """Gaussian radial basis functions with fixed, evenly-spaced centers."""

    def __init__(self, num_rbf: int = 20, cutoff: float = 5.0):
        super().__init__()
        self.register_buffer('centers', torch.linspace(0.0, cutoff, num_rbf))
        spacing = cutoff / (num_rbf - 1) if num_rbf > 1 else 1.0
        self.gamma = 1.0 / (2.0 * spacing ** 2)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        # dist: [E] -> [E, num_rbf]
        return torch.exp(-self.gamma * (dist.unsqueeze(-1) - self.centers) ** 2)


class CosineCutoff(nn.Module):
    """Smooth cosine cutoff envelope: 1 at d=0, 0 at d=cutoff."""

    def __init__(self, cutoff: float = 5.0):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        # dist: [E] -> [E]
        return 0.5 * (torch.cos(torch.pi * dist / self.cutoff) + 1.0) * (dist < self.cutoff).float()


class PaiNNMessage(nn.Module):
    """PaiNN message passing block using scatter operations.

    Computes scalar and vector messages from neighbor features,
    modulated by radial filters and cosine cutoff.

    Scalar: ds_i = sum_j filter_s(rbf_ij) * s_j
    Vector: dv_i = sum_j filter_v1(rbf_ij) * v_j
                 + sum_j filter_v2(rbf_ij) * s_j * unit_dir_ij
    """

    def __init__(self, hidden_dim: int, num_rbf: int, cutoff: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.filter_net = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim)
        )
        self.rbf = GaussianRBF(num_rbf, cutoff)
        self.cutoff_fn = CosineCutoff(cutoff)

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
        # each: [E, F]

        # Gather source features
        s_j = s[col]   # [E, F]
        v_j = v[col]   # [E, F, 3]

        # Scalar messages
        s_msg = filter_s * s_j  # [E, F]

        # Vector messages
        v_msg = (filter_v1.unsqueeze(-1) * v_j                           # [E, F, 3]
                 + (filter_v2 * s_j).unsqueeze(-1) * unit_dir.unsqueeze(1))  # [E, F, 3]

        # Aggregate to target nodes
        ds = scatter(s_msg, row, dim=0, dim_size=s.size(0), reduce='sum')   # [N, F]
        dv = scatter(v_msg, row, dim=0, dim_size=v.size(0), reduce='sum')   # [N, F, 3]

        return ds, dv


class FiLMGenerator(nn.Module):
    """Feature-wise Linear Modulation generator.

    Maps conditioning inputs (precision_settings + cheap_dft_scalars)
    to per-feature scale (gamma) and shift (beta) parameters.
    Gamma is initialized around 1.0 so FiLM starts as identity.
    """

    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim)
        )

    def forward(self, cond: torch.Tensor):
        # cond: [N, cond_dim] -> gamma: [N, F], beta: [N, F]
        out = self.net(cond)                        # [N, 2F]
        gamma, beta = out.chunk(2, dim=-1)          # each [N, F]
        gamma = gamma + 1.0                          # identity init
        return gamma, beta


class PaiNNUpdate(nn.Module):
    """PaiNN update block with FiLM conditioning.

    1. FiLM modulates scalar track: s_mod = s * gamma + beta
    2. Scalar-vector interaction via learned gating:
       s_new = a_ss * s_mod + a_sv * <v_t, v>
       v_new = a_vv * v_t
    """

    def __init__(self, hidden_dim: int, cond_dim: int, use_film: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_film = use_film

        # Linear transform on v before computing ||v|| and inner products
        self.v_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # MLP: [s_mod, ||v_t||] -> (a_ss, a_sv, a_vv)
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * hidden_dim)
        )

        # FiLM generator (only created when enabled)
        if use_film:
            self.film = FiLMGenerator(cond_dim, hidden_dim)

    def forward(self, s, v, cond):
        # s: [N, F], v: [N, F, 3], cond: [N, cond_dim]

        # 1. Transform v: apply linear over feature dim
        # v: [N, F, 3] -> transpose to [N, 3, F] -> linear -> [N, 3, F] -> transpose
        v_t = self.v_linear(v.transpose(1, 2)).transpose(1, 2)  # [N, F, 3]
        v_norm = v_t.norm(dim=-1)  # [N, F]

        # 2. FiLM modulation on scalars (skip when disabled)
        if self.use_film:
            gamma, beta = self.film(cond)   # each [N, F]
            s_mod = s * gamma + beta        # [N, F]
        else:
            s_mod = s

        # 3. Scalar-vector interaction
        sv_cat = torch.cat([s_mod, v_norm], dim=-1)                 # [N, 2F]
        update = self.update_mlp(sv_cat)                             # [N, 3F]
        a_ss, a_sv, a_vv = update.split(self.hidden_dim, dim=-1)    # each [N, F]

        # 4. Inner product <v_t, v>
        v_inner = (v_t * v).sum(dim=-1)  # [N, F]

        # 5. Scalar update
        ds = a_ss * s_mod + a_sv * v_inner  # [N, F]

        # 6. Vector update
        dv = a_vv.unsqueeze(-1) * v_t  # [N, F, 3]

        return ds, dv


class PaiNNInteraction(nn.Module):
    """One PaiNN interaction layer = Message + Update + residual + LayerNorm."""

    def __init__(self, hidden_dim: int, num_rbf: int, cutoff: float, cond_dim: int,
                 use_film: bool = True):
        super().__init__()
        self.message_block = PaiNNMessage(hidden_dim, num_rbf, cutoff)
        self.update_block = PaiNNUpdate(hidden_dim, cond_dim, use_film=use_film)
        self.scalar_norm = nn.LayerNorm(hidden_dim)

    def forward(self, s, v, edge_index, pos, cond):
        # s: [N, F], v: [N, F, 3], cond: [N, cond_dim]

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
