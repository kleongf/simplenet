"""
Unit tests for src/simplenet/dataset.py.

ArrayDataset/TensorDataset used to wrap the *entire* array in a single
Tensor at construction time and then try to slice into it per batch --
but Tensor has no __getitem__, so any indexing crashed immediately. Their
__getitem__ also only ever returned the data, silently dropping labels
entirely. train_test_split had the same underlying crash (it calls
dataset[i] internally) and, separately, returned plain Python lists
instead of Dataset-compatible objects, which Trainer can't batch-index.
"""

import numpy as np
import pytest

from simplenet.tensor import Tensor
from simplenet.dataset import Dataset, ArrayDataset, TensorDataset, Subset, train_test_split


class TestArrayDataset:
    def test_len(self):
        ds = ArrayDataset(np.random.randn(10, 3), np.random.randn(10, 1))
        assert len(ds) == 10

    def test_getitem_returns_data_and_labels(self):
        x = np.arange(20).reshape(10, 2).astype(np.float64)
        y = np.arange(10).astype(np.float64).reshape(10, 1)
        ds = ArrayDataset(x, y)
        xb, yb = ds[np.array([0, 2, 5])]
        assert isinstance(xb, Tensor) and isinstance(yb, Tensor)
        np.testing.assert_allclose(xb.data, x[[0, 2, 5]])
        np.testing.assert_allclose(yb.data, y[[0, 2, 5]])

    def test_getitem_single_index(self):
        x = np.random.randn(5, 3)
        y = np.random.randn(5, 1)
        ds = ArrayDataset(x, y)
        xb, yb = ds[2]
        np.testing.assert_allclose(xb.data, x[2])
        np.testing.assert_allclose(yb.data, y[2])

    def test_transform_applied_to_data_only(self):
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([10.0, 20.0, 30.0])
        ds = ArrayDataset(x, y, transform=lambda arr: arr * 2)
        xb, yb = ds[np.array([0, 1, 2])]
        np.testing.assert_allclose(xb.data, [[2.0], [4.0], [6.0]])
        np.testing.assert_allclose(yb.data, [10.0, 20.0, 30.0])


class TestTensorDataset:
    def test_len_and_getitem(self):
        x = Tensor(np.random.randn(6, 4))
        y = Tensor(np.random.randn(6, 1))
        ds = TensorDataset(x, y)
        assert len(ds) == 6
        xb, yb = ds[np.array([1, 3])]
        np.testing.assert_allclose(xb.data, x.data[[1, 3]])
        np.testing.assert_allclose(yb.data, y.data[[1, 3]])


class TestSubset:
    def test_len_matches_indices(self):
        ds = ArrayDataset(np.random.randn(10, 2), np.random.randn(10))
        sub = Subset(ds, np.array([0, 3, 7]))
        assert len(sub) == 3

    def test_getitem_delegates_through_indices(self):
        x = np.arange(10).reshape(10, 1).astype(np.float64)
        y = np.arange(10).astype(np.float64)
        ds = ArrayDataset(x, y)
        sub = Subset(ds, np.array([5, 1, 9]))
        xb, yb = sub[np.array([0, 1, 2])]
        np.testing.assert_allclose(xb.data.flatten(), [5.0, 1.0, 9.0])
        np.testing.assert_allclose(yb.data, [5.0, 1.0, 9.0])


class TestTrainTestSplit:
    def test_split_sizes(self):
        ds = ArrayDataset(np.random.randn(100, 2), np.random.randn(100))
        train_ds, test_ds = train_test_split(ds, test_size=0.3, random_state=0)
        assert len(train_ds) == 70
        assert len(test_ds) == 30

    def test_returns_dataset_compatible_objects(self):
        ds = ArrayDataset(np.random.randn(20, 2), np.random.randn(20))
        train_ds, test_ds = train_test_split(ds, test_size=0.25, random_state=0)
        assert isinstance(train_ds, Dataset)
        assert isinstance(test_ds, Dataset)
        # must support batch (array) indexing, not just single-item -- this
        # is exactly what Trainer relies on.
        xb, yb = train_ds[np.arange(len(train_ds))]
        assert xb.data.shape[0] == len(train_ds)

    def test_train_and_test_indices_do_not_overlap(self):
        x = np.arange(50).reshape(50, 1).astype(np.float64)
        ds = ArrayDataset(x, x.flatten())
        train_ds, test_ds = train_test_split(ds, test_size=0.2, random_state=1)
        train_vals = set(train_ds[np.arange(len(train_ds))][0].data.flatten().tolist())
        test_vals = set(test_ds[np.arange(len(test_ds))][0].data.flatten().tolist())
        assert train_vals.isdisjoint(test_vals)
        assert train_vals | test_vals == set(range(50))

    def test_reproducible_with_random_state(self):
        ds = ArrayDataset(np.arange(30).reshape(30, 1).astype(np.float64), np.arange(30))
        train_a, _ = train_test_split(ds, test_size=0.2, random_state=42)
        train_b, _ = train_test_split(ds, test_size=0.2, random_state=42)
        np.testing.assert_array_equal(train_a.indices, train_b.indices)
