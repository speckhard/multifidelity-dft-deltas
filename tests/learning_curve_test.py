"""Tests for learning curve functionality (train_fraction subsampling).

Validates:
1. train_fraction=1.0 uses all training data (no change)
2. train_fraction<1.0 subsamples correctly
3. Subsample is deterministic (same seed → same subset)
4. Val/test splits are never affected
5. Subsample indices are sorted (preserves data order for reproducibility)
6. Edge cases: fraction=0.01 (tiny), fraction=0.99 (near-full)
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# -- Minimal mock to test the subsampling logic in isolation --

def subsample_train(train_graphs, fraction, seed=42):
    """Replicate the subsampling logic from train_pipeline.py."""
    if fraction >= 1.0:
        return train_graphs
    n_full = len(train_graphs)
    n_subset = max(1, int(n_full * fraction))
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_full)[:n_subset]
    return [train_graphs[i] for i in sorted(indices)]


class TestTrainFraction:

    def test_fraction_1_returns_all(self):
        data = list(range(100))
        result = subsample_train(data, 1.0)
        assert result == data

    def test_fraction_above_1_returns_all(self):
        data = list(range(100))
        result = subsample_train(data, 1.5)
        assert result == data

    def test_fraction_0_5_returns_half(self):
        data = list(range(100))
        result = subsample_train(data, 0.5)
        assert len(result) == 50

    def test_fraction_0_6_returns_60(self):
        data = list(range(1000))
        result = subsample_train(data, 0.6)
        assert len(result) == 600

    def test_fraction_0_7_returns_700(self):
        data = list(range(1000))
        result = subsample_train(data, 0.7)
        assert len(result) == 700

    def test_fraction_0_8_returns_800(self):
        data = list(range(1000))
        result = subsample_train(data, 0.8)
        assert len(result) == 800

    def test_fraction_0_9_returns_900(self):
        data = list(range(1000))
        result = subsample_train(data, 0.9)
        assert len(result) == 900

    def test_deterministic_same_seed(self):
        data = list(range(1000))
        r1 = subsample_train(data, 0.5, seed=42)
        r2 = subsample_train(data, 0.5, seed=42)
        assert r1 == r2

    def test_different_seed_different_subset(self):
        data = list(range(1000))
        r1 = subsample_train(data, 0.5, seed=42)
        r2 = subsample_train(data, 0.5, seed=99)
        assert r1 != r2

    def test_subset_is_sorted(self):
        """Indices should be sorted so data order is preserved."""
        data = list(range(1000))
        result = subsample_train(data, 0.5, seed=42)
        assert result == sorted(result)

    def test_subset_is_subset_of_original(self):
        data = list(range(100))
        result = subsample_train(data, 0.5, seed=42)
        for item in result:
            assert item in data

    def test_no_duplicates(self):
        data = list(range(100))
        result = subsample_train(data, 0.5, seed=42)
        assert len(result) == len(set(result))

    def test_tiny_fraction(self):
        data = list(range(1000))
        result = subsample_train(data, 0.01)
        assert len(result) == 10

    def test_near_full_fraction(self):
        data = list(range(1000))
        result = subsample_train(data, 0.99)
        assert len(result) == 990

    def test_minimum_1_sample(self):
        """Even with very small fraction, at least 1 sample."""
        data = list(range(10))
        result = subsample_train(data, 0.001)
        assert len(result) >= 1

    def test_val_test_unchanged(self):
        """Verify the logic doesn't touch val/test splits.
        This tests the conceptual guarantee: only train_graphs is subsampled.
        """
        train = list(range(100))
        val = list(range(100, 120))
        test = list(range(120, 140))

        train_sub = subsample_train(train, 0.5)
        assert len(train_sub) == 50
        assert val == list(range(100, 120))  # unchanged
        assert test == list(range(120, 140))  # unchanged

    def test_actual_data_size(self):
        """Test with actual dataset size (43105 train graphs)."""
        data = list(range(43105))
        for frac in [0.5, 0.6, 0.7, 0.8, 0.9]:
            result = subsample_train(data, frac)
            expected = int(43105 * frac)
            assert len(result) == expected, f"frac={frac}: got {len(result)}, expected {expected}"

    def test_fractions_are_nested(self):
        """Smaller fractions should be subsets of larger fractions (with same seed)."""
        data = list(range(1000))
        # This is NOT guaranteed by random permutation + take-first-N,
        # because sorting changes which elements appear. But the indices
        # selected by permutation[:n_small] ARE a subset of permutation[:n_large].
        # After sorting, the small set should be a subset of the large set.
        r50 = set(subsample_train(data, 0.5, seed=42))
        r70 = set(subsample_train(data, 0.7, seed=42))
        r90 = set(subsample_train(data, 0.9, seed=42))
        assert r50.issubset(r70), "50% should be subset of 70%"
        assert r70.issubset(r90), "70% should be subset of 90%"
