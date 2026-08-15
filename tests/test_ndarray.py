"""
test_array.py

Test suite for the Array class (pure-Python N-dimensional array).

Run with:
    pytest ./tests/test_ndarray.py -v
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
# zeros / ones / full
# ---------------------------------------------------------------------------

class TestZerosOnesFull:
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

    def test_full_shape_and_values(self):
        arr = Array.full(7, (2, 3))
        assert arr.shape == (2, 3)
        assert arr.data == [7, 7, 7, 7, 7, 7]

    def test_full_1d(self):
        arr = Array.full(-1.5, (4,))
        assert arr.data == [-1.5, -1.5, -1.5, -1.5]


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
# broadcast_shapes
# ---------------------------------------------------------------------------

class TestBroadcastShapes:
    def test_equal_shapes(self):
        assert Array.broadcast_shapes((2, 3), (2, 3)) == (2, 3)

    def test_vector_against_matrix(self):
        assert Array.broadcast_shapes((3,), (2, 3)) == (2, 3)

    def test_matrix_against_vector(self):
        # order shouldn't matter
        assert Array.broadcast_shapes((2, 3), (3,)) == (2, 3)

    def test_row_broadcast(self):
        assert Array.broadcast_shapes((1, 3), (2, 3)) == (2, 3)

    def test_column_broadcast(self):
        assert Array.broadcast_shapes((2, 1), (2, 3)) == (2, 3)

    def test_both_have_size_one_dims(self):
        assert Array.broadcast_shapes((2, 1), (1, 3)) == (2, 3)

    def test_incompatible_shapes_raise(self):
        with pytest.raises(ValueError):
            Array.broadcast_shapes((2,), (3,))

    def test_incompatible_matrix_shapes_raise(self):
        with pytest.raises(ValueError):
            Array.broadcast_shapes((2, 3), (2, 4))


# ---------------------------------------------------------------------------
# broadcast_to
# ---------------------------------------------------------------------------

class TestBroadcastTo:
    def test_vector_to_matrix_tiles_rows(self):
        v = Array([10, 20, 30], shape=(3,))
        b = v.broadcast_to((2, 3))
        assert b.shape == (2, 3)
        assert b._unflatten(b.data, b.shape) == [[10, 20, 30], [10, 20, 30]]

    def test_column_to_matrix_tiles_columns(self):
        c = Array([[1], [2]], shape=(2, 1))
        b = c.broadcast_to((2, 3))
        assert b._unflatten(b.data, b.shape) == [[1, 1, 1], [2, 2, 2]]

    def test_row_to_matrix_tiles_rows(self):
        r = Array([[1, 2, 3]], shape=(1, 3))
        b = r.broadcast_to((2, 3))
        assert b._unflatten(b.data, b.shape) == [[1, 2, 3], [1, 2, 3]]

    def test_same_shape_returns_self_no_copy(self):
        a = Array([[1, 2], [3, 4]])
        b = a.broadcast_to((2, 2))
        assert b is a

    def test_incompatible_shape_raises(self):
        a = Array([1, 2], shape=(2,))
        with pytest.raises(ValueError):
            a.broadcast_to((3,))


# ---------------------------------------------------------------------------
# unbroadcast
# ---------------------------------------------------------------------------

class TestUnbroadcast:
    def test_unbroadcast_vector_from_matrix(self):
        # (3,) was broadcast to (2,3)
        grad = Array([[1, 2, 3], [4, 5, 6]])

        result = grad.unbroadcast((3,))

        assert result.shape == (3,)
        assert result.data == [5, 7, 9]

    def test_unbroadcast_row_vector(self):
        # (1,3) was broadcast to (2,3)
        grad = Array([[1, 2, 3], [4, 5, 6]])

        result = grad.unbroadcast((1, 3))

        assert result.shape == (1, 3)
        assert result.data == [5, 7, 9]

    def test_unbroadcast_column_vector(self):
        # (2,1) was broadcast to (2,3)
        grad = Array([[1, 2, 3], [4, 5, 6]])

        result = grad.unbroadcast((2, 1))

        assert result.shape == (2, 1)
        assert result.data == [6, 15]

    def test_unbroadcast_scalar_like_array(self):
        # (1,1) was broadcast to (2,3)
        grad = Array([[1, 2, 3], [4, 5, 6]])

        result = grad.unbroadcast((1, 1))

        assert result.shape == (1, 1)
        assert result.data == [21]

    def test_unbroadcast_extra_leading_dimensions_3d(self):
        # (3,) was broadcast to (2, 4, 3)
        grad = Array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ]
        )

        result = grad.unbroadcast((3,))

        assert result.shape == (3,)
        assert result.data == [22, 26, 30]

    def test_unbroadcast_3d_to_2d(self):
        # (2,3) was broadcast to (4,2,3)
        grad = Array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
                [[13, 14, 15], [16, 17, 18]],
                [[19, 20, 21], [22, 23, 24]],
            ]
        )

        result = grad.unbroadcast((2, 3))

        assert result.shape == (2, 3)
        assert result.data == [40, 44, 48, 52, 56, 60]

    def test_unbroadcast_multiple_broadcast_dimensions(self):
        # (1,3,1) was broadcast to (2,3,4)
        grad = Array(
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                ],
                [
                    [13, 14, 15, 16],
                    [17, 18, 19, 20],
                    [21, 22, 23, 24],
                ],
            ]
        )

        result = grad.unbroadcast((1, 3, 1))

        assert result.shape == (1, 3, 1)
        assert result.data == [68, 100, 132]

    def test_unbroadcast_same_shape_returns_same_values(self):
        grad = Array([[1, 2], [3, 4]])

        result = grad.unbroadcast((2, 2))

        assert result.shape == (2, 2)
        assert result.data == [1, 2, 3, 4]

    def test_unbroadcast_result_can_be_broadcast_back(self):
        # This verifies the important relationship:
        #
        # broadcast -> gradient -> unbroadcast
        #
        # The resulting shape should be the original operand shape.
        grad = Array([[1, 2, 3], [4, 5, 6]])

        result = grad.unbroadcast((3,))

        assert result.broadcast_to((2, 3)).shape == (2, 3)
        assert result.broadcast_to((2, 3)).data == [
            5, 7, 9,
            5, 7, 9,
        ]


# ---------------------------------------------------------------------------
# Arithmetic operators — equal shapes, scalars, and broadcasting
# ---------------------------------------------------------------------------

class TestArithmeticEqualShapes:
    def test_add(self):
        a = Array([[1, 2], [3, 4]])
        b = Array([[10, 20], [30, 40]])
        assert (a + b)._unflatten((a + b).data, (a + b).shape) == [[11, 22], [33, 44]]

    def test_sub(self):
        a = Array([[10, 20], [30, 40]])
        b = Array([[1, 2], [3, 4]])
        assert (a - b)._unflatten((a - b).data, (a - b).shape) == [[9, 18], [27, 36]]

    def test_mul(self):
        a = Array([[1, 2], [3, 4]])
        b = Array([[10, 20], [30, 40]])
        assert (a * b)._unflatten((a * b).data, (a * b).shape) == [[10, 40], [90, 160]]

    def test_truediv(self):
        a = Array([[10.0, 20.0]])
        b = Array([[2.0, 4.0]])
        assert (a / b).data == [5.0, 5.0]

    def test_incompatible_shapes_raise(self):
        with pytest.raises(ValueError):
            Array([1, 2], shape=(2,)) + Array([1, 2, 3], shape=(3,))


class TestArithmeticScalars:
    def test_array_plus_scalar(self):
        a = Array([[1, 2], [3, 4]])
        assert (a + 5).data == [6, 7, 8, 9]

    def test_scalar_plus_array(self):
        a = Array([[1, 2], [3, 4]])
        assert (5 + a).data == [6, 7, 8, 9]

    def test_array_minus_scalar(self):
        a = Array([[1, 2], [3, 4]])
        assert (a - 1).data == [0, 1, 2, 3]

    def test_scalar_minus_array_not_commutative(self):
        a = Array([[1, 2], [3, 4]])
        assert (10 - a).data == [9, 8, 7, 6]

    def test_array_times_scalar(self):
        a = Array([[1, 2], [3, 4]])
        assert (a * 2).data == [2, 4, 6, 8]

    def test_array_divided_by_scalar(self):
        a = Array([[2.0, 4.0]])
        assert (a / 2).data == [1.0, 2.0]

    def test_scalar_divided_by_array_not_commutative(self):
        a = Array([[2.0, 4.0]])
        assert (16 / a).data == [8.0, 4.0]

    def test_neg(self):
        a = Array([1.0, -2.0, 3.0])
        assert (-a).data == [-1.0, 2.0, -3.0]

    def test_pow(self):
        a = Array([2.0, 3.0, 4.0])
        assert (a ** 2).data == [4.0, 9.0, 16.0]


class TestArithmeticBroadcasting:
    def test_matrix_plus_row_vector(self):
        a = Array([[1, 2, 3], [4, 5, 6]])          # (2, 3)
        bias = Array([10, 20, 30], shape=(3,))      # (3,)
        result = a + bias
        assert result.shape == (2, 3)
        assert result._unflatten(result.data, result.shape) == [[11, 22, 33], [14, 25, 36]]

    def test_matrix_plus_column_vector(self):
        col = Array([[1], [2]], shape=(2, 1))
        mat = Array([[1, 2, 3], [4, 5, 6]])
        result = col + mat
        assert result.shape == (2, 3)
        assert result._unflatten(result.data, result.shape) == [[2, 3, 4], [6, 7, 8]]

    def test_broadcast_multiply(self):
        a = Array([[1, 2], [3, 4]])
        row = Array([10, 100], shape=(2,))
        result = a * row
        assert result._unflatten(result.data, result.shape) == [[10, 200], [30, 400]]


# ---------------------------------------------------------------------------
# Elementwise math functions
# ---------------------------------------------------------------------------

class TestMathFunctions:
    def test_exp(self):
        import math
        a = Array([0.0, 1.0, 2.0])
        result = a.exp()
        assert result.allclose(Array([math.exp(0.0), math.exp(1.0), math.exp(2.0)]))

    def test_log(self):
        import math
        a = Array([1.0, math.e, math.e ** 2])
        result = a.log()
        assert result.allclose(Array([0.0, 1.0, 2.0]))

    def test_log_nonpositive_raises(self):
        a = Array([1.0, -1.0])
        with pytest.raises(ValueError):
            a.log()

    def test_sqrt(self):
        a = Array([4.0, 9.0, 16.0])
        result = a.sqrt()
        assert result.allclose(Array([2.0, 3.0, 4.0]))

    def test_sqrt_negative_raises(self):
        a = Array([-1.0])
        with pytest.raises(ValueError):
            a.sqrt()


# ---------------------------------------------------------------------------
# Reduction operations — sum / mean, full and axis-aware
# ---------------------------------------------------------------------------

class TestSumMean:
    def test_full_sum(self):
        a = Array([[1, 2, 3], [4, 5, 6]])
        assert a.sum().data == [21]
        assert a.sum().shape == (1,)

    def test_full_mean(self):
        a = Array([[1, 2, 3], [4, 5, 6]])
        assert a.mean().data == [3.5]

    def test_sum_axis_0(self):
        a = Array([[1, 2, 3], [4, 5, 6]])
        result = a.sum(axis=0)
        assert result.shape == (3,)
        assert result.data == [5, 7, 9]

    def test_sum_axis_1(self):
        a = Array([[1, 2, 3], [4, 5, 6]])
        result = a.sum(axis=1)
        assert result.shape == (2,)
        assert result.data == [6, 15]

    def test_sum_negative_axis(self):
        a = Array([[1, 2, 3], [4, 5, 6]])
        # axis=-1 should behave like axis=1 for a 2D array
        assert a.sum(axis=-1).data == a.sum(axis=1).data

    def test_sum_axis_keepdims(self):
        a = Array([[1, 2, 3], [4, 5, 6]])
        result = a.sum(axis=0, keepdims=True)
        assert result.shape == (1, 3)
        assert result.data == [5, 7, 9]

    def test_mean_axis_1(self):
        a = Array([[1, 2, 3], [4, 5, 6]])
        result = a.mean(axis=1)
        assert result.shape == (2,)
        assert result.data == [2.0, 5.0]

    def test_mean_axis_0_keepdims(self):
        a = Array([[1, 2, 3], [4, 5, 6]])
        result = a.mean(axis=0, keepdims=True)
        assert result.shape == (1, 3)
        assert result.data == [2.5, 3.5, 4.5]

    def test_sum_axis_result_broadcasts_back_against_original(self):
        # A common pattern in gradient code: reduce along an axis, then
        # broadcast the result back to add against the original shape.
        a = Array([[1, 2, 3], [4, 5, 6]])
        col_sums = a.sum(axis=0, keepdims=True)  # shape (1, 3)
        # should not raise, since (1,3) broadcasts against (2,3)
        combined = a + col_sums
        assert combined.shape == (2, 3)


# ---------------------------------------------------------------------------
# Matrix multiplication — 2D and batched 3D
# ---------------------------------------------------------------------------

class TestMatmul2D:
    def test_basic_matmul(self):
        a = Array([[1, 2], [3, 4]])
        b = Array([[5, 6], [7, 8]])
        result = a.matmul(b)
        assert result.shape == (2, 2)
        assert result._unflatten(result.data, result.shape) == [[19, 22], [43, 50]]

    def test_matmul_operator(self):
        a = Array([[1, 2], [3, 4]])
        b = Array([[5, 6], [7, 8]])
        assert (a @ b)._unflatten((a @ b).data, (a @ b).shape) == [[19, 22], [43, 50]]

    def test_non_square_matmul(self):
        a = Array([[1, 2, 3], [4, 5, 6]])       # (2, 3)
        b = Array([[7, 8], [9, 10], [11, 12]])   # (3, 2)
        result = a.matmul(b)
        assert result.shape == (2, 2)

    def test_incompatible_inner_dims_raise(self):
        a = Array([[1, 2]], shape=(1, 2))
        b = Array([[1, 2, 3]], shape=(1, 3))
        with pytest.raises(ValueError):
            a.matmul(b)


class TestMatmulBatched3D:
    def test_batch_slice_extracts_correct_data(self):
        a = Array([[1, 2], [3, 4]])
        b = Array([[5, 6], [7, 8]])
        stacked = Array.stack([a, b])
        assert stacked.shape == (2, 2, 2)
        assert stacked._batch_slice(0)._unflatten(
            stacked._batch_slice(0).data, stacked._batch_slice(0).shape
        ) == [[1, 2], [3, 4]]
        assert stacked._batch_slice(1)._unflatten(
            stacked._batch_slice(1).data, stacked._batch_slice(1).shape
        ) == [[5, 6], [7, 8]]

    def test_stack_empty_raises(self):
        with pytest.raises(ValueError):
            Array.stack([])

    def test_stack_mismatched_shapes_raises(self):
        a = Array([[1, 2], [3, 4]])
        b = Array([1, 2, 3], shape=(3,))
        with pytest.raises(ValueError):
            Array.stack([a, b])

    def test_batched_matmul_matches_individual_2d_matmuls(self):
        a = Array([[1, 2], [3, 4]])
        b = Array([[5, 6], [7, 8]])
        expected = a.matmul(b)

        batch_a = Array.stack([a, a])
        batch_b = Array.stack([b, b])
        result = batch_a.matmul(batch_b)

        assert result.shape == (2, 2, 2)
        assert result._batch_slice(0).allclose(expected)
        assert result._batch_slice(1).allclose(expected)

    def test_batched_matmul_via_operator(self):
        a = Array([[1, 2], [3, 4]])
        batch_a = Array.stack([a, a])
        batch_b = Array.stack([a, a])
        result = batch_a @ batch_b
        assert result.shape == (2, 2, 2)

    def test_batch_size_mismatch_raises(self):
        a = Array([[1, 2], [3, 4]])
        batch_1 = Array.stack([a])
        batch_2 = Array.stack([a, a])
        with pytest.raises(ValueError):
            batch_1.matmul(batch_2)

    def test_mixed_2d_and_3d_raises(self):
        a = Array([[1, 2], [3, 4]])
        batch_a = Array.stack([a, a])
        with pytest.raises(ValueError):
            a.matmul(batch_a)


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