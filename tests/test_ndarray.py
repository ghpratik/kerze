"""
test_array.py

Test suite for the Array class (pure-Python N-dimensional array).

Run with:
    pytest test_array.py -v
"""

import pytest
from kerze.ndarray import Array  # rename import to match your actual filename


# ---------------------------------------------------------------------------
# Construction: nested input, shape inference
# ---------------------------------------------------------------------------

class TestConstructionFromNested:
    def test_infers_shape_2d(self):
        arr = Array([[1, 2, 3], [4, 5, 6]])
        assert arr.shape == (2, 3)

    def test_flattens_2d_data(self):
        arr = Array([[1, 2, 3], [4, 5, 6]])
        assert arr.data == [1, 2, 3, 4, 5, 6]

    def test_infers_shape_1d(self):
        arr = Array([1, 2, 3])
        assert arr.shape == (3,)
        assert arr.data == [1, 2, 3]

    def test_infers_shape_3d(self):
        arr = Array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        assert arr.shape == (2, 2, 2)
        assert arr.data == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_computes_strides_2d(self):
        arr = Array([[1, 2, 3], [4, 5, 6]])
        assert arr.strides == (3, 1)

    def test_computes_strides_3d(self):
        arr = Array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        assert arr.strides == (4, 2, 1)


class TestConstructionWithExplicitShape:
    def test_flat_data_with_matching_shape(self):
        arr = Array([1, 2, 3, 4, 5, 6], shape=(2, 3))
        assert arr.data == [1, 2, 3, 4, 5, 6]
        assert arr.shape == (2, 3)

    def test_nested_data_with_matching_shape_gets_flattened(self):
        # This is the bug we fixed earlier: nested data + explicit shape
        # must still flatten, not store the nested list as-is.
        arr = Array([[2, 4, 1], [3, 6, 2]], shape=(2, 3))
        assert arr.data == [2, 4, 1, 3, 6, 2]

    def test_mismatched_shape_raises(self):
        with pytest.raises(ValueError):
            Array([1, 2, 3, 4, 5], shape=(2, 3))  # 5 elements, shape needs 6

    def test_mismatched_shape_raises_for_nested_input_too(self):
        with pytest.raises(ValueError):
            Array([[1, 2, 3], [4, 5, 6]], shape=(3, 3))  # 6 elements, shape needs 9

    def test_reshape_to_different_valid_shape(self):
        # e.g. flat 9 elements reshaped to (3,3) instead of inferred (whatever)
        arr = Array([1, 2, 3, 4, 5, 6, 7, 8, 9], shape=(3, 3))
        assert arr.shape == (3, 3)
        assert arr.get(2, 2) == 9


# ---------------------------------------------------------------------------
# get / set
# ---------------------------------------------------------------------------

class TestGetSet:
    def test_get_2d(self):
        arr = Array([[1, 2, 3], [4, 5, 6]])
        assert arr.get(0, 0) == 1
        assert arr.get(0, 2) == 3
        assert arr.get(1, 1) == 5
        assert arr.get(1, 2) == 6

    def test_get_1d(self):
        arr = Array([10, 20, 30])
        assert arr.get(0) == 10
        assert arr.get(2) == 30

    def test_get_3d(self):
        arr = Array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        assert arr.get(0, 0, 0) == 1
        assert arr.get(1, 1, 1) == 8
        assert arr.get(1, 0, 1) == 6

    def test_set_updates_value(self):
        arr = Array([[1, 2, 3], [4, 5, 6]])
        arr.set(99, 1, 1)
        assert arr.get(1, 1) == 99

    def test_set_does_not_affect_other_elements(self):
        arr = Array([[1, 2, 3], [4, 5, 6]])
        arr.set(99, 0, 0)
        assert arr.data == [99, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# unflatten (round-trip with construction)
# ---------------------------------------------------------------------------

class TestUnflatten:
    def test_round_trip_2d(self):
        original = [[2, 4, 1], [3, 6, 2]]
        arr = Array(original)
        assert arr._unflatten(arr.data, arr.shape) == original

    def test_round_trip_1d(self):
        original = [1, 2, 3]
        arr = Array(original)
        assert arr._unflatten(arr.data, arr.shape) == original

    def test_round_trip_3d(self):
        original = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        arr = Array(original)
        assert arr._unflatten(arr.data, arr.shape) == original

    def test_round_trip_after_explicit_shape_construction(self):
        arr = Array([[2, 4, 1], [3, 6, 2]], shape=(2, 3))
        assert arr._unflatten(arr.data, arr.shape) == [[2, 4, 1], [3, 6, 2]]


# ---------------------------------------------------------------------------
# reshape
# ---------------------------------------------------------------------------

class TestReshape:
    def test_reshape_preserves_data_order(self):
        arr = Array([1, 2, 3, 4, 5, 6], shape=(2, 3))
        reshaped = arr.reshape((3, 2))
        assert reshaped.shape == (3, 2)
        assert reshaped.data == [1, 2, 3, 4, 5, 6]  # same flat order

    def test_reshape_get_values_correct(self):
        arr = Array([1, 2, 3, 4, 5, 6], shape=(2, 3))
        reshaped = arr.reshape((3, 2))
        # flat = [1,2,3,4,5,6] -> as (3,2): [[1,2],[3,4],[5,6]]
        assert reshaped.get(0, 0) == 1
        assert reshaped.get(1, 0) == 3
        assert reshaped.get(2, 1) == 6

    def test_reshape_to_incompatible_size_raises(self):
        arr = Array([1, 2, 3, 4, 5, 6], shape=(2, 3))
        with pytest.raises(ValueError):
            arr.reshape((4, 4))

    def test_reshape_to_1d(self):
        arr = Array([[1, 2], [3, 4]])
        flat = arr.reshape((4,))
        assert flat.shape == (4,)
        assert flat.data == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# transpose
# ---------------------------------------------------------------------------

class TestTranspose:
    def test_transpose_shape_reversed(self):
        arr = Array([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
        t = arr.transpose()
        assert t.shape == (3, 2)

    def test_transpose_strides_reversed(self):
        arr = Array([[1, 2, 3], [4, 5, 6]])  # strides (3, 1)
        t = arr.transpose()
        assert t.strides == (1, 3)

    def test_transpose_values_correct(self):
        arr = Array([[1, 2, 3], [4, 5, 6]])
        t = arr.transpose()
        # original[i][j] should equal transposed[j][i]
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                assert arr.get(i, j) == t.get(j, i)

    def test_transpose_is_zero_copy(self):
        arr = Array([[1, 2, 3], [4, 5, 6]])
        t = arr.transpose()
        # Same underlying flat buffer object, not a copy.
        assert t.data is arr.data

    def test_transpose_of_transpose_is_original_shape(self):
        arr = Array([[1, 2, 3], [4, 5, 6]])
        tt = arr.transpose().transpose()
        assert tt.shape == arr.shape
        assert tt.strides == arr.strides


# ---------------------------------------------------------------------------
# zeros / ones
# ---------------------------------------------------------------------------

class TestZerosOnes:
    def test_zeros_shape_and_values(self):
        arr = Array.zeros((2, 3))
        assert arr.shape == (2, 3)
        assert arr.data == [0.0] * 6

    def test_ones_shape_and_values(self):
        arr = Array.ones((2, 3))
        assert arr.shape == (2, 3)
        assert arr.data == [1.0] * 6

    def test_zeros_1d(self):
        arr = Array.zeros((5,))
        assert arr.data == [0.0, 0.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# allclose
# ---------------------------------------------------------------------------

class TestAllclose:
    def test_identical_arrays_are_close(self):
        a = Array([[1, 2], [3, 4]])
        b = Array([[1, 2], [3, 4]])
        assert a.allclose(b)

    def test_slightly_different_within_tolerance(self):
        a = Array([[1.0, 2.0]])
        b = Array([[1.0000001, 2.0000001]])
        assert a.allclose(b)

    def test_different_values_not_close(self):
        a = Array([[1, 2], [3, 4]])
        b = Array([[1, 2], [3, 5]])
        assert not a.allclose(b)

    def test_different_shapes_not_close(self):
        a = Array([1, 2, 3])
        b = Array([[1, 2, 3]])
        assert not a.allclose(b)

    def test_custom_tolerance(self):
        a = Array([1.0])
        b = Array([1.1])
        assert not a.allclose(b, tol=1e-6)
        assert a.allclose(b, tol=0.2)


# ---------------------------------------------------------------------------
# ndim / size properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_ndim_2d(self):
        arr = Array([[1, 2, 3], [4, 5, 6]])
        assert arr.ndim == 2

    def test_ndim_1d(self):
        arr = Array([1, 2, 3])
        assert arr.ndim == 1

    def test_ndim_3d(self):
        arr = Array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        assert arr.ndim == 3

    def test_size_2d(self):
        arr = Array([[1, 2, 3], [4, 5, 6]])
        assert arr.size == 6

    def test_size_matches_len_data(self):
        arr = Array([[1, 2, 3], [4, 5, 6]])
        assert arr.size == len(arr.data)


# ---------------------------------------------------------------------------
# __repr__ (basic sanity, not exact string matching)
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_does_not_crash(self):
        arr = Array([[1, 2], [3, 4]])
        assert "shape=(2, 2)" in repr(arr)

    def test_repr_contains_data(self):
        arr = Array([[1, 2], [3, 4]])
        assert "[[1, 2], [3, 4]]" in repr(arr)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])