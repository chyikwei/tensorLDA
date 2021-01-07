import numpy as np

from itertools import permutations
from numpy.linalg import pinv

from six.moves import xrange
from numpy.testing import assert_equal
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_raises_regexp

from tensor_lda.utils.tensor_utils import (_check_1d_vector,
                                           rank_1_tensor_3d,
                                           khatri_rao_prod,
                                           tensor_3d_permute,
                                           tensor_3d_from_matrix_vector,
                                           tensor_3d_from_vector_matrix,
                                           tensor_3d_prod)


def test_check_1d_vectors():
    # test check_1d_vectors function
    rng = np.random.RandomState(0)

    dim = rng.randint(50, 100)
    # check (dim, )
    a = _check_1d_vector(np.ones((dim,)))
    assert_equal(1, len(a.shape))
    assert_equal(dim, a.shape[0])

    # check (dim, 1)
    b = _check_1d_vector(np.ones((dim, 1)))
    assert_equal(1, len(b.shape))
    assert_equal(dim, b.shape[0])

    # check (dim, 2)
    c = np.ones((dim, 2))
    assert_raises_regexp(ValueError, r"^Vector is not 1-d array:",
                         _check_1d_vector, c)


def test_create_3d_rank_1_tensor_simple():
    # test create_3d_rank_1_tensor

    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    c = np.array([7, 8, 9])
    result = np.array(
        [[28, 35, 42, 32, 40, 48, 36, 45, 54],
         [56, 70, 84, 64, 80, 96, 72, 90, 108],
         [84, 105, 126, 96, 120, 144, 108, 135, 162]])

    tensor = rank_1_tensor_3d(a, b, c)
    assert_array_equal(result, tensor)


def test_create_3d_rank_1_tensor_symmetric():
    rng = np.random.RandomState(0)
    dim = rng.randint(20, 25)
    v = rng.rand(dim)
    tensor = rank_1_tensor_3d(v, v, v)

    for i in xrange(dim):
        for j in xrange(i, dim):
            for k in xrange(j, dim):
                true_val = v[i] * v[j] * v[k]
                # check all permutation have same values
                for perm in permutations([i, j, k]):
                    tensor_val = tensor[perm[0], (dim * perm[2]) + perm[1]]
                    assert_almost_equal(true_val, tensor_val)


def test_create_3d_rank_1_tensor_random():
    # test create_3d_rank_1_tensor with random values
    rng = np.random.RandomState(0)

    dim = rng.randint(20, 25)
    a = rng.rand(dim)
    b = rng.rand(dim)
    c = rng.rand(dim)

    tensor = rank_1_tensor_3d(a, b, c)
    assert_equal(2, len(tensor.shape))
    assert_equal(dim, tensor.shape[0])
    assert_equal(dim * dim, tensor.shape[1])

    for i in xrange(dim):
        for j in xrange(dim):
            for k in xrange(dim):
                true_val = a[i] * b[j] * c[k]
                tensor_val = tensor[i, (dim * k) + j]
                assert_almost_equal(true_val, tensor_val)


def test_khatri_rao_prod():
    # test khatri_rao_prod
    rng = np.random.RandomState(0)

    dim_row_a = rng.randint(10, 20)
    dim_row_b = rng.randint(10, 20)
    dim_row_prod = dim_row_a * dim_row_b
    dim_col = rng.randint(10, 20)

    a = rng.rand(dim_row_a, dim_col)
    b = rng.rand(dim_row_b, dim_col)
    prod = khatri_rao_prod(a, b)

    assert_equal(2, len(prod.shape))
    assert_equal(dim_row_prod, prod.shape[0])
    assert_equal(dim_col, prod.shape[1])

    for i in xrange(dim_row_prod):
        for j in xrange(dim_col):
            true_val = a[i // dim_row_b, j] * b[i % dim_row_b, j]
            assert_almost_equal(true_val, prod[i, j])


def test_khatri_rao_properties():
    # test properties of Kron & Khatri-Rao product
    # check eq(2) in reference[?]
    rng = np.random.RandomState(0)

    dim_row_a = rng.randint(20, 50)
    dim_row_b = rng.randint(20, 50)
    dim_row_c = rng.randint(20, 50)
    dim_col = rng.randint(20, 40)

    a = rng.rand(dim_row_a, dim_col)
    b = rng.rand(dim_row_b, dim_col)

    # eq(2) line 1
    c = rng.rand(dim_col, dim_row_a)
    d = rng.rand(dim_col, dim_row_b)
    kr_prod_1 = np.dot(np.kron(a, b), np.kron(d, c))
    kr_prod_2 = np.kron(np.dot(a, d), np.dot(b, c))
    assert_array_almost_equal(kr_prod_1, kr_prod_2)

    # eq(2) line 2
    kr_inv = pinv(np.kron(a, b))
    kr_inv_reconstruct = np.kron(pinv(a), pinv(b))
    assert_array_almost_equal(kr_inv, kr_inv_reconstruct)

    # eq(2) line 3
    c = rng.rand(dim_row_c, dim_col)
    prod_1 = khatri_rao_prod(khatri_rao_prod(a, b), c)
    prod_2 = khatri_rao_prod(a, khatri_rao_prod(b, c))
    assert_array_almost_equal(prod_1, prod_2)

    # eq(2) line 4
    prod = khatri_rao_prod(a, b)
    result = np.dot(a.T, a) * np.dot(b.T, b)
    assert_array_almost_equal(np.dot(prod.T, prod), result)

    # eq(2) line 5
    prod_inv = pinv(prod)
    result_inv = pinv(result)
    reconstruct_inv = np.dot(result_inv, prod.T)
    assert_array_almost_equal(prod_inv, reconstruct_inv)


def test_tensor_3d_permute():
    rng = np.random.RandomState(0)

    dim1 = rng.randint(10, 20)
    dim2 = rng.randint(10, 20)
    dim3 = rng.randint(10, 20)

    tensor = rng.rand(dim1, (dim2 * dim3))

    #dim1 = 2
    #dim2 = 3
    #dim3 = 4

    #mtx = np.arange(6).reshape(2, 3)
    #vector = np.array([7, 8, 9, 10])

    #tensor = tensor_3d_from_matrix_vector(mtx, vector)

    # test (2, 3, 1) mode
    permute_2_3_1 = tensor_3d_permute(tensor, (dim1, dim2, dim3), a=2, b=3, c=1)
    assert_equal(dim2, permute_2_3_1.shape[0])
    assert_equal(dim3 * dim1, permute_2_3_1.shape[1])

    #print tensor
    #print permute_2_3_1

    for i1 in xrange(dim2):
        for i2 in xrange(dim3):
            for i3 in xrange(dim1):
                val_permute = permute_2_3_1[i1, (dim3 * i3) + i2]
                val_origin = tensor[i3, (dim2 * i2) + i1]
                assert_equal(val_permute, val_origin)

    # TODO: test other mode


def test_tensor_3d_from_matrix_vector():
    rng = np.random.RandomState(0)

    dim1 = rng.randint(10, 20)
    dim2 = rng.randint(10, 20)
    dim3 = rng.randint(10, 20)

    mtx = rng.rand(dim1, dim2)
    vector = rng.rand(dim3)

    tensor = tensor_3d_from_matrix_vector(mtx, vector)
    for i in xrange(dim1):
        for j in xrange(dim2):
            for k in xrange(dim3):
                val_true = mtx[i, j] * vector[k]
                val = tensor[i, (dim2 * k) + j]
                assert_equal(val_true, val)


def test_tensor_3d_from_vector_matrix():
    rng = np.random.RandomState(3)

    dim1 = rng.randint(10, 20)
    dim2 = rng.randint(10, 20)
    dim3 = rng.randint(10, 20)

    vector = rng.rand(dim1)
    mtx = rng.rand(dim2, dim3)

    tensor = tensor_3d_from_vector_matrix(vector, mtx)
    for i in xrange(dim1):
        for j in xrange(dim2):
            for k in xrange(dim3):
                val_true = vector[i] * mtx[j, k]
                val = tensor[i, (dim2 * k) + j]
                assert_equal(val_true, val)


def test_tensor_3d_prod():

    def get_prod_val(t, a, b, c, i1, i2, i3):
        t1, t2, t3 = t.shape
        val = 0.0
        for j1 in xrange(t1):
            for j2 in xrange(t2):
                for j3 in xrange(t3):
                    val += (t[j1, j2, j3] * a[j1, i1] * b[j2, i2] * c[j3, i3])
        return val

    rng = np.random.RandomState(3)

    tensor = rng.rand(3, 4, 5)
    a = rng.rand(3, 6)
    b = rng.rand(4, 7)
    c = rng.rand(5, 8)

    t2 = tensor_3d_prod(tensor, a, b, c)
    assert_equal(6, t2.shape[0])
    assert_equal(7, t2.shape[1])
    assert_equal(8, t2.shape[2])

    for i in xrange(6):
        for j in xrange(7):
            for k in xrange(8):
                val_true = get_prod_val(tensor, a, b, c, i, j, k)
                assert_almost_equal(val_true, t2[i, j, k])
