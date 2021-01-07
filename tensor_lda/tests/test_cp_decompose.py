import numpy as np

from numpy.testing import assert_equal
from nose.tools import assert_greater_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_raises_regexp

from six.moves import xrange

from scipy.stats import ortho_group
from tensor_lda.cp_decompose import (tensor_reconstruct,
                                     cp_als,
                                     cp_tensor_power_method)
from tensor_lda.utils.tensor_utils import rank_1_tensor_3d


def _create_symmertic_tensor(consts, vectors, random_state):
    dim = consts.shape[0]
    tensor = np.zeros((dim, dim * dim))
    for i in xrange(dim):
        vector = vectors[:, i]
        tensor += (consts[i] * rank_1_tensor_3d(vector, vector, vector))
    return tensor


def test_cp_als():
    # test decompose 3D symmertic tensor from random orthogonal vectors

    rng = np.random.RandomState(2)
    dim = rng.randint(10, 15)
    # create orthogonal vectors
    consts = 10. + 2. * rng.rand(dim)
    consts.sort()
    vectors = ortho_group.rvs(dim, random_state=rng)
    # tensor = sum of rank one tensors
    tensor = _create_symmertic_tensor(consts, vectors, rng)

    # running als decompose
    lambdas, a, b, c = cp_als(tensor, dim, 100, 1000, 1e-8, rng)
    lambda_order = np.argsort(lambdas)

    #print consts
    #print lambdas[lambda_order]
    assert_equal(dim, len(lambdas))
    assert_equal((dim, dim), a.shape)
    assert_equal((dim, dim), b.shape)
    assert_equal((dim, dim), c.shape)
    #print("lambda %r" % lambdas)
    #print("a %r" % a)
    #print("b %r" % b)
    #print("c %r" % c)

    # check matrix are orthogonal
    for matrix in [a, b, c]:
        for col1 in xrange(dim):
            v = matrix[:, col1]
            assert_almost_equal(1., np.dot(v, v))
            for col2 in range(col1 + 1, dim):
                #print("lambda2 %.3f" % lambdas[col2])
                #print("%d, %d" % (col1, col2))
                assert_almost_equal(0., np.dot(v, matrix[:, col2]))

    # reconstruct
    reconstruct_tensor = tensor_reconstruct(lambdas, a, b, c)
    assert_array_almost_equal(tensor, reconstruct_tensor, decimal=6)

    # vector should be the smae
    assert_array_almost_equal(np.abs(a), np.abs(b))
    assert_array_almost_equal(np.abs(a), np.abs(c))

    # compare each vector
    lambda_order = np.argsort(lambdas)
    for i in xrange(dim):
        reconstruct_idx = lambda_order[i]
        assert_almost_equal(consts[i], lambdas[reconstruct_idx])
        v = vectors[:, i]
        rebuid_v = a[:, reconstruct_idx]
        # there can bw a sign diff
        if (v[0] * rebuid_v[0]) < 0.:
            rebuid_v *= -1.
        assert_array_almost_equal(v, rebuid_v)


def test_cp_tensor_power_method():

    rng = np.random.RandomState(10)
    dim = rng.randint(10, 15)
    # create orthogonal vectors
    consts = 10. + 5. * rng.rand(dim)
    consts.sort()
    vectors = ortho_group.rvs(dim, random_state=rng)
    # tensor = sum of rank one tensors
    tensor = _create_symmertic_tensor(consts, vectors, rng)

    # running als decompose
    lambdas, decompose_vecs, n_iter = cp_tensor_power_method(tensor, dim, 10, 100, rng)
    assert_greater_equal(n_iter, 1)
    lambda_order = np.argsort(lambdas)

    #print consts
    #print lambdas[lambda_order]
    assert_equal(dim, len(lambdas))
    assert_equal((dim, dim), decompose_vecs.shape)

    for col1 in xrange(dim):
        v = decompose_vecs[:, col1]
        assert_almost_equal(1., np.dot(v, v))
        for col2 in range(col1 + 1, dim):
            #print("lambda2 %.3f" % lambdas[col2])
            #print("%d, %d" % (col1, col2))
            assert_almost_equal(0., np.dot(v, decompose_vecs[:, col2]))

    reconstruct_tensor = tensor_reconstruct(
        lambdas, decompose_vecs, decompose_vecs, decompose_vecs)
    assert_array_almost_equal(tensor, reconstruct_tensor, decimal=6)

    # compare each vector
    lambda_order = np.argsort(lambdas)
    for i in xrange(dim):
        reconstruct_idx = lambda_order[i]
        assert_almost_equal(consts[i], lambdas[reconstruct_idx])
        assert_array_almost_equal(
            vectors[:, i], decompose_vecs[:, reconstruct_idx])
