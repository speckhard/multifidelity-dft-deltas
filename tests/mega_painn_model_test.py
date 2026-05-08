"""Tests for MegaPaiNN surrogate model.

Tests cover: forward shapes, backward pass, scalar invariance under rotation,
embedding output, FiLM toggle, and multi-graph batching.
"""

import os
import sys

import pytest
import torch
from torch_geometric.data import Batch, Data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from modelling.gnn.mega_painn_model import MegaPaiNN  # noqa: E402


# ---- Helpers ----

def make_model(**overrides):
    """Create a small MegaPaiNN for testing."""
    defaults = dict(
        num_layers=2,
        hidden_dim=16,
        num_precision_settings=12,
        max_z=10,
        num_rbf=8,
        cutoff=5.0,
        embedding_dim=32,
        use_film=False,
    )
    defaults.update(overrides)
    return MegaPaiNN(**defaults)


def make_graph(num_atoms=3, seed=0):
    """Create a single PyG Data graph (no batch dim yet)."""
    rng = torch.manual_seed(seed)
    pos = torch.randn(num_atoms, 3)
    z = torch.randint(1, 9, (num_atoms, 1))

    # Simple fully-connected edges
    src = []
    dst = []
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    prec = torch.zeros(1, 12)
    prec[0, 10] = 1.0
    prec[0, 11] = 8.0

    return Data(
        x=pos,
        z=z,
        edge_index=edge_index,
        precision_settings=prec,
    )


def make_batch(num_graphs=2, atoms_per_graph=3):
    """Create a batched PyG graph."""
    graphs = [make_graph(atoms_per_graph, seed=i) for i in range(num_graphs)]
    return Batch.from_data_list(graphs)


# ---- Forward shapes ----

class TestForwardShapes:

    def test_single_graph(self):
        """Single graph produces correct output shapes."""
        model = make_model()
        batch = make_batch(num_graphs=1, atoms_per_graph=4)
        out = model(batch)

        assert out['e_form'].shape == (1,)
        assert out['e_gap'].shape == (1,)
        assert out['embedding'].shape == (1, 32)

    def test_multi_graph_batch(self):
        """Batched graphs produce correct output shapes."""
        model = make_model()
        batch = make_batch(num_graphs=5, atoms_per_graph=3)
        out = model(batch)

        assert out['e_form'].shape == (5,)
        assert out['e_gap'].shape == (5,)
        assert out['embedding'].shape == (5, 32)

    def test_different_embedding_dim(self):
        """Custom embedding_dim is respected."""
        model = make_model(embedding_dim=64)
        batch = make_batch(num_graphs=2)
        out = model(batch)
        assert out['embedding'].shape == (2, 64)

    def test_return_is_dict(self):
        """Forward returns a dict with expected keys."""
        model = make_model()
        batch = make_batch(num_graphs=1)
        out = model(batch)
        assert isinstance(out, dict)
        assert set(out.keys()) == {'e_form', 'e_gap', 'embedding'}


# ---- Backward pass ----

class TestBackward:

    def test_gradients_flow(self):
        """Loss backward propagates gradients to all parameters."""
        model = make_model()
        batch = make_batch(num_graphs=2)
        out = model(batch)

        loss = out['e_form'].sum() + out['e_gap'].sum() + out['embedding'].sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f'No gradient for {name}'
                assert not torch.all(param.grad == 0), \
                    f'Zero gradient for {name}'

    def test_embedding_gradient(self):
        """Embedding head receives gradients."""
        model = make_model()
        batch = make_batch(num_graphs=2)
        out = model(batch)

        loss = out['embedding'].sum()
        loss.backward()

        assert model.embedding_head.weight.grad is not None
        assert not torch.all(model.embedding_head.weight.grad == 0)


# ---- Scalar invariance ----

class TestInvariance:

    def test_rotation_invariance(self):
        """Scalar outputs (e_form, e_gap) are invariant under rotation."""
        model = make_model()
        model.eval()

        batch1 = make_batch(num_graphs=2, atoms_per_graph=4)

        # Random rotation matrix (SO(3))
        q, r = torch.linalg.qr(torch.randn(3, 3))
        # Ensure proper rotation (det = +1)
        q = q * torch.sign(torch.diag(r)).unsqueeze(0)
        if torch.det(q) < 0:
            q[:, 0] *= -1

        # Rotate positions
        batch2 = make_batch(num_graphs=2, atoms_per_graph=4)
        batch2.x = batch1.x @ q.T

        with torch.no_grad():
            out1 = model(batch1)
            out2 = model(batch2)

        torch.testing.assert_close(
            out1['e_form'], out2['e_form'], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            out1['e_gap'], out2['e_gap'], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            out1['embedding'], out2['embedding'], atol=1e-4, rtol=1e-4)

    def test_translation_invariance(self):
        """Outputs are invariant under translation of all positions."""
        model = make_model()
        model.eval()

        batch1 = make_batch(num_graphs=1, atoms_per_graph=3)
        batch2 = make_batch(num_graphs=1, atoms_per_graph=3)
        batch2.x = batch1.x + torch.tensor([10.0, -5.0, 3.0])

        with torch.no_grad():
            out1 = model(batch1)
            out2 = model(batch2)

        torch.testing.assert_close(
            out1['e_form'], out2['e_form'], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            out1['e_gap'], out2['e_gap'], atol=1e-4, rtol=1e-4)


# ---- FiLM toggle ----

class TestFiLM:

    def test_film_off_runs(self):
        """Model runs with use_film=False (default)."""
        model = make_model(use_film=False)
        batch = make_batch(num_graphs=2)
        out = model(batch)
        assert out['e_form'].shape == (2,)

    def test_film_on_runs(self):
        """Model runs with use_film=True."""
        model = make_model(use_film=True)
        batch = make_batch(num_graphs=2)
        out = model(batch)
        assert out['e_form'].shape == (2,)

    def test_film_on_more_params(self):
        """FiLM=True adds parameters vs FiLM=False."""
        model_off = make_model(use_film=False)
        model_on = make_model(use_film=True)

        n_off = sum(p.numel() for p in model_off.parameters())
        n_on = sum(p.numel() for p in model_on.parameters())
        assert n_on > n_off


# ---- Edge cases ----

class TestEdgeCases:

    def test_single_atom_graph(self):
        """Handles graph with 1 atom (no edges)."""
        model = make_model()
        data = Data(
            x=torch.randn(1, 3),
            z=torch.tensor([[6]]),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            precision_settings=torch.zeros(1, 12),
            batch=torch.tensor([0]),
        )
        out = model(data)
        assert out['e_form'].shape == (1,)
        assert out['e_gap'].shape == (1,)

    def test_large_z(self):
        """Handles high atomic numbers (within max_z)."""
        model = make_model(max_z=120)
        data = Data(
            x=torch.randn(2, 3),
            z=torch.tensor([[79], [92]]),  # Au, U
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            precision_settings=torch.zeros(1, 12),
            batch=torch.tensor([0, 0]),
        )
        out = model(data)
        assert out['e_form'].shape == (1,)
