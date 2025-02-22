import unittest

import numpy as np


class DynamicNumpyArray:
    def __init__(self, dim: int):
        self.dim = dim
        self._allocated_map: list[bool] = [False] * 5
        self._raw = np.zeros((5, dim))
        self.public = self._raw[0:0, :]

    def _update_public(self):
        self.public = self._raw[self._allocated_map, :]

    def __len__(self):
        return self.public.shape[0]

    def __index__(self, idx: int):
        res = self.get(idx)
        if res is None:
            raise IndexError
        return res

    def __getitem__(self, item):
        res = self.get(item)
        if res is None:
            raise IndexError
        return res

    def extend_memory(self, num: int = 10):
        self._allocated_map.extend([False] * num)
        self._raw = np.append(self._raw, np.zeros((num, self.dim)), axis=0)

    def insert(self, value):
        if len(value) != self.dim:
            raise ValueError("Invalid value length")

        if self._raw.shape[0] == len(self):
            self.extend_memory()

        idx = self._allocated_map.index(False)
        self._raw[idx, :] = value
        self._allocated_map[idx] = True

        self._update_public()

    def remove(self, idx: int):
        self._allocated_map[idx] = False
        self._update_public()

    def get(self, idx: int):
        if abs(idx) >= len(self):
            return None
        return self.public[idx, :]


class TestDynamicNumpyArray(unittest.TestCase):
    def setUp(self):
        self.array = DynamicNumpyArray(dim=3)

    def test_initialization(self):
        self.assertEqual(self.array.dim, 3)
        self.assertEqual(len(self.array), 0)
        self.assertTrue(np.array_equal(self.array.public, np.zeros((0, 3))))

    def test_insert(self):
        self.array.insert([1, 2, 3])
        self.assertEqual(len(self.array), 1)
        self.assertTrue(np.array_equal(self.array.public, np.array([[1, 2, 3]])))

    def test_extend_memory(self):
        for i in range(5):
            self.array.insert([i, i + 1, i + 2])
        self.assertEqual(len(self.array), 5)
        self.array.insert([5, 6, 7])
        self.assertEqual(len(self.array), 6)
        self.assertEqual(self.array._raw.shape[0], 15)

    def test_remove(self):
        self.array.insert([1, 2, 3])
        self.array.insert([4, 5, 6])
        self.array.remove(0)
        self.assertEqual(len(self.array), 1)
        self.assertTrue(np.array_equal(self.array.public, np.array([[4, 5, 6]])))

    def test_get(self):
        self.array.insert([1, 2, 3])
        self.array.insert([4, 5, 6])
        self.assertTrue(np.array_equal(self.array.get(0), np.array([1, 2, 3])))
        self.assertTrue(np.array_equal(self.array.get(1), np.array([4, 5, 6])))

    def test_invalid_insert(self):
        with self.assertRaises(ValueError):
            self.array.insert([1, 2])

    def test_invalid_get(self):
        self.assertIsNone(self.array.get(0))
