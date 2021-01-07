import numpy as np
import scipy.sparse as sp

from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_array_almost_equal

from tensor_lda.tensor_lda import TensorLDA


def test_estimator():
    """Test estimator"""
    lda = TensorLDA(n_components=10, alpha0=1.)
    return check_estimator(lda)


def test_tensor_lda_simple():
    rng = np.random.RandomState(0)

    n_features = 500
    n_samples = rng.randint(100, 200)
    n_topics = 50
    alpha0 = 1.
    doc_word_mtx = rng.randint(0, 3, size=(n_samples, n_features))
    doc_word_mtx = sp.csr_matrix(doc_word_mtx)

    lda = TensorLDA(n_components=50, alpha0=alpha0)
    lda.fit(doc_word_mtx)

    assert_almost_equal(alpha0, lda.alpha_.sum())
    assert_array_almost_equal(np.ones(n_topics), lda.components_.sum(axis=1))
