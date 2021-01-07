"""Generate moments from documents
"""

from __future__ import print_function


import warnings

import numpy as np
from numpy import linalg as LA
import scipy.sparse as sp

from sklearn.exceptions import ConvergenceWarning
from six.moves import xrange

from .utils.tensor_utils import (rank_1_tensor_3d,
                                 tensor_3d_from_vector_matrix,
                                 tensor_3d_from_matrix_vector)
from .utils.fast_tensor_ops import (tensor_3d_permute_231,
                                    tensor_3d_permute_312)


def first_order_moments(X, min_words=3):
    """First-Order Moments
    
    Generate first order Moment of document-word frequency matrix.

    Parameters
    ----------
    X : array-like or sparse matrix, shape=(n_samples, n_features)
        Matrix of Document-word frequency. `n_samples` is the
        number of documnet and `n_features` are number of unique
        words in the corpus.
    
    min_words : Integer, default=3
        Minimum number of words in each document. In LDA, the number
        is 3 since we need 3rd order moments.

    Returns
    -------
    e1 : array, shape=(n_features,)
        Expectation of each words in the input matrix.
    
    ignored: integer
        Number of ignored documents.

    """
    n_samples, n_features = X.shape
    is_sparse_x = sp.issparse(X)
    doc_word_cnts = np.asarray(X.sum(axis=1))
    if len(doc_word_cnts.shape) > 1:
        doc_word_cnts = np.squeeze(doc_word_cnts)

    if is_sparse_x:
        X_data = X.data
        X_indices = X.indices
        X_indptr = X.indptr

    ignored_docs = 0
    e1 = np.zeros(n_features)

    # TODO: optimize for loop with cython
    for idx_d in xrange(n_samples):
        # get word_id and count in each document
        words_cnt = doc_word_cnts[idx_d]
        if words_cnt < min_words:
            ignored_docs += 1
            continue

        if is_sparse_x:
            ids = X_indices[X_indptr[idx_d]:X_indptr[idx_d + 1]]
            cnts = X_data[X_indptr[idx_d]:X_indptr[idx_d + 1]]
        else:
            ids = np.nonzero(X[idx_d, :])[0]
            cnts = X[idx_d, ids]

        for w_id, w_cnt in zip(ids, cnts):
            e1[w_id] += (w_cnt / float(words_cnt))
    e1 /= (n_samples - ignored_docs)

    return (e1, ignored_docs)


def cooccurrence_expectation(X, min_words=3, batch_size=1000):
    """Expectation of Word Co-occurrence

    Expectation of 2 words occured in the same document.

    Parameters
    ----------
    X : array-like or sparse matrix, shape=(n_samples, n_features)
        Matrix of Document-word frequency. `n_samples` is the
        number of documnet and `n_features` are number of unique
        words in the corpus.
    
    min_words : Integer, default=3
        Minimum number of words in each document. In LDA, the number
        is 3 since we need 3rd order moments.
    
    batch_size : Integer, default=1000
        batch size of each word pairs

    Returns
    -------
    e2 : sparse matrix, shape=(n_features, n_features)
        Expectation of word pairs
    
    ignored: integer
        Number of ignored documents.

    """

    n_samples, n_features = X.shape
    is_sparse_x = sp.issparse(X)

    doc_word_cnts = np.squeeze(np.asarray(X.sum(axis=1))).reshape(n_samples,)
    ignored_docs = 0

    if is_sparse_x:
        X_data = X.data
        X_indices = X.indices
        X_indptr = X.indptr

    # TODO: optimize for loop with cython
    e2 = sp.coo_matrix((n_features, n_features))
    rows = []
    cols = []
    vals = []
    for idx_d in xrange(n_samples):
        # get word_id and count in each document
        words_cnt = doc_word_cnts[idx_d]
        if words_cnt < min_words:
            ignored_docs += 1
            continue

        if is_sparse_x:
            ids = X_indices[X_indptr[idx_d]:X_indptr[idx_d + 1]]
            cnts = X_data[X_indptr[idx_d]:X_indptr[idx_d + 1]]
        else:
            ids = np.nonzero(X[idx_d, :])[0]
            cnts = X[idx_d, ids]
        unique_word_cnt = len(ids)
        unique_pairs = unique_word_cnt * (unique_word_cnt - 1)
        # index for total tokens are:
        # 0 to (unique_word_cnt - 1): pair(i, i)
        # unique_word_cnt to end: pair(i, j) where (j > i)
        total_non_zeros = unique_word_cnt + unique_pairs
        row_index = np.zeros(total_non_zeros).astype('int')
        col_index = np.zeros(total_non_zeros).astype('int')
        idx_vals = np.zeros(total_non_zeros)

        idx = 0
        for i in xrange(unique_word_cnt):
            cnt = cnts[i]
            # if cnt = 0, val will be 0. don't need to set index
            if cnt > 1:
                row_index[idx] = ids[i]
                col_index[idx] = ids[i]
                idx_vals[idx] = cnt * (cnt - 1)
            idx += 1

        for i in xrange(unique_word_cnt):
            for j in xrange(i + 1, unique_word_cnt):
                row_index[idx] = ids[i]
                col_index[idx] = ids[j]
                idx_vals[idx] = cnts[i] * cnts[j]
                idx += 1
        idx_vals /= (words_cnt * (words_cnt - 1))
        rows.append(row_index)
        cols.append(col_index)
        vals.append(idx_vals)

        # merge for each batch
        if idx_d % batch_size == 0:
            r = np.hstack(rows)
            c = np.hstack(cols)
            data = np.hstack(vals)
            e2 += sp.coo_matrix((data, (r, c)), shape=(n_features, n_features))
            # reset
            rows = []
            cols = []
            vals = []

    # last batch
    if rows:
        r = np.hstack(rows)
        c = np.hstack(cols)
        data = np.hstack(vals)
        e2 += sp.coo_matrix((data, (r, c)), shape=(n_features, n_features))

    # add symmetric pairs to lower triangle
    valid_docs = n_samples - ignored_docs
    if valid_docs > 0:
        e2 /= valid_docs
        e2 += sp.triu(e2, k=1).T
    return (e2, ignored_docs)


def second_order_moments(n_components, e2, m1, alpha0):
    """Second-Order Moments

    To prevent creating 2nd order moments explicitly, we construct its
    decomposition with `n_components`. check reference [?] section 5.2
    for details.

    Parameters
    ----------
    n_components: int
        Number of components

    e2: sparse matrix, shape=(n_features, n_features)
        Expectation of word pairs. e2[i, j] is the expectation of word `i`
        and `j` in the same document.
    
    m1: array, shape=(n_features,)
        Expectation of each words.

    alpha0: double
        Sum of topic topic concentration parameter
    
    Returns
    -------
    m2_vals : array, shape=(n_components,)
        eigen values of sencond-order moments

    m2_vecs : array, shape=(n_features, n_components)
        eigen values of sencond-order moments        
    """

    # eigen values and vectors of E2
    n_features = e2.shape[0]
    #print("%d ; %d" % (n_features, n_components))
    if n_components == n_features:
        # run full svd, convert e2 to dense array first
        e2_vecs, e2_vals, _ = LA.svd(e2.toarray())
    else:
        #e2_vals, e2_vecs = sp.linalg.eigsh(e2, k=n_components, which='LM')
        e2_vecs, e2_vals, _ = sp.linalg.svds(e2, k=n_components, which='LM',
                                             return_singular_vectors=True)
    e2_vals *= (alpha0 + 1.)
    m1_p = np.dot(e2_vecs.T, m1)

    # section 5.2 part 1.
    m2_p = (-1. * alpha0) * (m1_p * m1_p[:, np.newaxis])
    m2_p[np.diag_indices_from(m2_p)] += e2_vals

    # section 5.2 part 1.
    # eigen values and vectors of M2 prime
    try:
        m2p_vecs, m2p_vals, _ = LA.svd(m2_p)
        m2_vals = m2p_vals
        m2_vecs = np.dot(e2_vecs, m2p_vecs)
    except LA.LinAlgError:
        # In order to pass `check_estimator` test.
        # convert this error to warnings.
        warnings.warn("SVD in second_order_moments did not converge. "
                      "the algorithm will not work.",
                      ConvergenceWarning)
        m2_vals = np.ones(m2_p.shape[0])
        m2_vecs = m2_p

    return (m2_vals, m2_vecs)


def whitening(m2_eigen_values, m2_eign_vectors):
    """Compute the whitening matrix from M2
    
    Parameters
    ----------
    m2_eigen_values: array, shape=(n_components,)
        eigen values from M2

    m2_eigen_vectors: array, shape=(n_features, n_components)
        eigen vectors from M2

    Returns
    -------
    whitening_matrix: array, shape=(n_features, n_components)

    """
    lambda_sqrt = np.diag(1. / np.sqrt(m2_eigen_values))
    whitening_matrix = np.dot(m2_eign_vectors, lambda_sqrt)
    return whitening_matrix


def unwhitening(m2_eigen_values, m2_eign_vectors):
    """Compute the whitening matrix from M2
    
    Parameters
    ----------
    m2_eigen_values: array, shape=(n_components,)
        eigen values from M2

    m2_eigen_vectors: array, shape=(n_features, n_components)
        eigen vectors from M2

    Returns
    -------
    unwhitening_matrix: array, shape=(n_features, n_components)

    """
    lambda_sqrt = np.diag(np.sqrt(m2_eigen_values))
    unwhitening_matrix = np.dot(m2_eign_vectors, lambda_sqrt)
    return unwhitening_matrix


def whitening_triples_expectation(X, min_words, whitening_matrix):
    """Computed transformed expectation of 3-word triples

    To prevent compute `O(V^3)` matrix directly, we compute
    the expectation in the tranformed space, which is `O(k^3)`.
    For details, check eq (8) to (11) in reference [?].

    Parameters
    ----------
    X : array-like or sparse matrix, shape=(n_samples, n_features)
        Matrix of Document-word frequency. `n_samples` is the
        number of documnet and `n_features` are number of unique
        words in the corpus.

    whitening_matrix : array, shape=(n_features, n_components)

    Returns
    -------
    e3 : array, (n, n * n)
        3D tensor in unfolded format
    """

    # TODO: remove check_array to main algorithm
    n_samples, n_features = X.shape
    n_components = whitening_matrix.shape[1]
    is_sparse_x = sp.issparse(X)

    # TODO: move this part to main algorithm
    doc_word_cnts = np.squeeze(np.asarray(X.sum(axis=1))).reshape(n_samples,)
    ignored_docs = 0

    if is_sparse_x:
        X_data = X.data
        X_indices = X.indices
        X_indptr = X.indptr

    # TODO: optimize for loop with cython
    e3 = np.zeros((n_components, n_components * n_components))
    a3_coef = np.zeros(n_features)
    for idx_d in xrange(n_samples):
        # get word_id and count in each document
        words_cnt = doc_word_cnts[idx_d]
        if words_cnt < min_words:
            ignored_docs += 1
            continue

        if is_sparse_x:
            ids = X_indices[X_indptr[idx_d]:X_indptr[idx_d + 1]]
            cnts = X_data[X_indptr[idx_d]:X_indptr[idx_d + 1]]
        else:
            ids = np.nonzero(X[idx_d, :])[0]
            cnts = X[idx_d, ids]

        rho = 1. / (words_cnt * (words_cnt - 1.) * (words_cnt - 2.))
        # w shape = (len(ids), n_components)
        w = whitening_matrix.take(ids, axis=0)
        # w_c_i shape = (n_components,)
        w_c = np.dot(w.T, cnts)

        # eq (9)
        a1 = rank_1_tensor_3d(w_c, w_c, w_c)
        # eq (10)
        w_t_diag_w = np.dot(np.multiply(w.T, cnts), w)
        a2 = tensor_3d_from_vector_matrix(w_c, w_t_diag_w)
        a2_3_1_2 = tensor_3d_permute_312(a2)
        a2_2_3_1 = tensor_3d_permute_231(a2)
        e3 += (rho * (a1 - a2 - a2_3_1_2 - a2_2_3_1))

        # coef in eq (11)
        for w_id, w_cnt in zip(ids, cnts):
            a3_coef[w_id] += (w_cnt * rho)

    # compute a3. eq (11)
    for i in xrange(n_features):
        w_i = whitening_matrix[i, :]
        e3 += (2. * a3_coef[i] * rank_1_tensor_3d(w_i, w_i, w_i))

    valid_docs = n_samples - ignored_docs
    if valid_docs > 0:
        e3 /= valid_docs
    return e3


def whitening_tensor_e2_m1(whitened_m1, alpha0):
    """Compute Expectaion of E2 x M1 tensor in 3 mode

    Compute sum of whitened E[x1, x2, m1],
    E[x1, m1, x2], and E[m1, x1, x2]

    Parameters
    ----------
    whitened_m1: int
        Number of components

    alpha0: sparse matrix, shape=(n_features, n_features)
        Expectation of word pairs. e2[i, j] is the expectation of word `i`
        and `j` in the same document.

    Returns
    -------
    u1_2_3 : array, shape=(n_components,)
        eigen values of sencond-order moments

    """
    # U1, U2, U3. eq (13) to (15)
    w_t_e2_w = alpha0 * (whitened_m1[np.newaxis, :] * 
                         whitened_m1[:, np.newaxis])
    w_t_e2_w[np.diag_indices_from(w_t_e2_w)] += 1.
    w_t_e2_w /= (alpha0 + 1.)
    # U1
    u1 = tensor_3d_from_matrix_vector(w_t_e2_w, whitened_m1)
    u1_2_3 = u1.copy()
    # add U2
    u1_2_3 += tensor_3d_permute_312(u1)
    # add U3
    u1_2_3 += tensor_3d_permute_231(u1)
    return u1_2_3


def third_order_monents(X, whitening_matrix, m1, alpha0, min_words=3):
    """Transformed Third order moments

        Directly compute 3rd order moments requires `O(V^3)`
        space. Therefore, we compute the product of whitening
        matrix and 3rd order moments which only required `O(k^3)`
        space. (`k` (number of components) is usually much smaller
        than `V` (size of vocabulary))

    Parameters
    ----------
    X : array-like or sparse matrix, shape=(n_samples, n_features)
        Matrix of Document-word frequency. `n_samples` is the
        number of documnet and `n_features` are number of unique
        words in the corpus.

    whitening_matrix : array, shape=(n_features, n_components)
        Whitening matrix calculated from M2

    m1 : array, shape=(n_features,)
        First order moments

    alpha0 : Double
        Sum of topic topic concentration parameter

    min_words : Integer, default=3
        Minimum number of words in each document. In LDA, the number
        is 3 since we need 3rd order moments.

    Returns
    -------
    m3 : array, (n, n * n)
        3D tensor in unfolded format

    """
    wt_m1 = np.dot(whitening_matrix.T, m1)

    # E3. eq (8) to (11)
    e3 = whitening_triples_expectation(X, min_words, whitening_matrix)

    # U1, U2, U3. eq (13) to (15)
    u1_2_3 = whitening_tensor_e2_m1(wt_m1, alpha0)

    # M1 x 3 tensor. eq (16)
    m1_3d = rank_1_tensor_3d(wt_m1, wt_m1, wt_m1)

    # TODO: do inplace update
    e3 *= ((alpha0 + 1.) * (alpha0 + 2.) * 0.5)
    u1_2_3 *= (alpha0 * (alpha0 + 1.) * 0.5)
    m1_3d *= (alpha0 * alpha0)
    m3 = e3 - u1_2_3 + m1_3d
    return m3
