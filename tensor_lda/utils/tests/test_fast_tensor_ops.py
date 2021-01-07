import numpy as np
from sklearn.utils._testing import assert_array_almost_equal

from tensor_lda.utils.tensor_utils import tensor_3d_permute
from tensor_lda.utils.fast_tensor_ops import (tensor_3d_permute_231,
                                              tensor_3d_permute_312)


def test_tensor_3d_permute_231():
    rng = np.random.RandomState(0)

    dim = rng.randint(10, 20)
    tensor = rng.rand(dim, (dim * dim))

    permute_2_3_1 = tensor_3d_permute(tensor, (dim, dim, dim), a=2, b=3, c=1)
    fast_permute_2_3_1 = tensor_3d_permute_231(tensor)
    assert_array_almost_equal(permute_2_3_1, fast_permute_2_3_1)


def test_tensor_3d_permute_312():
    rng = np.random.RandomState(1)

    dim = rng.randint(10, 20)
    tensor = rng.rand(dim, (dim * dim))

    permute_3_1_2 = tensor_3d_permute(tensor, (dim, dim, dim), a=3, b=1, c=2)
    fast_permute_3_1_2 = tensor_3d_permute_312(tensor)
    assert_array_almost_equal(permute_3_1_2, fast_permute_3_1_2)
