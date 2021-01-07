"""Utility functions for tensor operations
"""

import numpy as np

from six.moves import xrange


def _check_1d_vector(vector):
    """Check 1D vector shape

    Check 1D vector shape. array with shape
    [n, 1] or [n, ] are accepted. Will return
    a 1 dimension vector.

    Parameters
    ----------
    vector : array (n,) or (n, 1)
        rank one vector
    
    Returns
    -------
    vector : array, (n,)
    """

    v_shape = vector.shape
    if len(v_shape) == 1:
        return vector
    elif len(v_shape) == 2 and v_shape[1] == 1:
        return vector.reshape(v_shape[0],)
    else:
        raise ValueError("Vector is not 1-d array: shape %s" % str(v_shape))


def _check_square_matrix(matrix):
    """Check 2D matrix shape

    Check 1D vector shape. array with shape
    [n, 1] or [n, ] are accepted. Will return
    a 1 dimension vector.

    Parameters
    ----------
    matrix : (n, n)
        rank one vector
    
    Returns
    -------
    matrix : array, (n, n)
    """
    m_shape = matrix.shape

    if len(m_shape) == 2:
        if m_shape[0] != m_shape[1]:
            raise ValueError("matrix is not square: shape %s" % str(m_shape))
        return matrix
    else:
        raise ValueError("matrix is not 2-d array: shape %s" % str(m_shape))


def rank_1_tensor_3d(a, b, c):
    """Generate a 3-D tensor from 3 1-D vectors

    Generate a 3D tensor from 3 rank one vectors 
    `a`, `b`, and `c`. The returned 3-D tensor is
    in unfolded format.

    Parameters
    ----------
    a : array, shape (n,)
        first rank one vector

    b : array, shape (n,)
        second rank one vector

    c : array, shape (n,)
        thrid rank one vector
    
    Returns
    -------
    tensor:  array, (n, n * n)
        3D tensor in unfolded format. element
        (i, j, k) will map to (i, (n * k) + j)
    """

    a = _check_1d_vector(a)
    b = _check_1d_vector(b)
    c = _check_1d_vector(c)

    dim = a.shape[0]
    # check dimension
    if (dim != b.shape[0]) or (dim != c.shape[0]):
        raise ValueError("Vector dimension mismatch: (%d, %d, %d)" %
                         (dim, b.shape[0], c.shape[0]))

    outter = b[:, np.newaxis] * c[:, np.newaxis].T
    tensor = a[:, np.newaxis] * outter.ravel(order='F')[np.newaxis, :]
    return tensor


def tensor_3d_from_vector_matrix(a, b):
    """Generate 3-D tensor from 1-D vector and 2-D matrix

    Generate a 3D tensor from a 1-D vector `a` and 2-D
    matrix `b`. The returned 3-D tensor is
    in unfolded format.
    
    Parameters
    ----------
    a : array, shape (m,)
        1-D vector

    b : 2-D array, shape (n, p)
        2-D matrix

    Returns
    -------
    tensor:  array, (m, n * p)
        3D tensor in unfolded format.

    """
    a = _check_1d_vector(a)
    tensor = a[:, np.newaxis] * b.ravel(order='F')[np.newaxis, :]
    return tensor


def tensor_3d_from_matrix_vector(b, a):
    """Generate 3-D tensor from 2-D matrix and 1-D vector

    This function is similar to `tensor_3d_from_vector_matrix`
    function. The only difference is the first argument is 2-D
    matrix and the second element is 1-D vector.

    Parameters
    ----------
    b : array, shape (m, n)
        2-D matrix

    a : array, shape (p,)
        vector

    Returns
    -------
    tensor : array, shape (m, n * p)
        3D tensor in unfolded format.

    """
    len_a = a.shape[0]
    n_col = b.shape[1]
    tensor = np.tile(b, len_a)
    for i in xrange(len_a):
        col_from = n_col * i
        col_to = n_col * (i+1)
        tensor[:, col_from:col_to] *= a[i]
    return tensor


def tensor_3d_permute(tensor, tensor_shape, a, b, c):
    """Permute the mode of a 3-D tensor
    
    This is a slow implementation to generate 3-D tensor
    permutations.

    Parameters
    ----------
    tensor : 2D array, shape (n, m * k)
        3D tensor in unfolded format

    tensor_shape : int triple
        Shape of the tensor. Since tensor is in
        unfolded format. We need it's real format
        to calculate permutation.

    a : int, {1, 2, 3}
        new first index
    }

    b : int, {1, 2, 3}
        new second index

    c : int, {1, 2, 3}
        new thrid order index

    Return
    ------
    permuted_tensor:  2D array
        Permuted tensor, element (i_1, i_2, i_3) in
        the permuted tensor will be element
        (i_a, i_b, i_c) in the original tensor
    """

    # TODO: check parameter
    a_idx = a - 1
    b_idx = b - 1
    c_idx = c - 1
    # TODO: move this part to cython loop
    n_col = tensor_shape[1]
    dim1 = tensor_shape[a_idx]
    dim2 = tensor_shape[b_idx]
    dim3 = tensor_shape[c_idx]

    permuted_tensor = np.empty((dim1, dim2 * dim3))
    old_idx = np.zeros(3).astype('int32')
    for i in xrange(dim1):
        for j in xrange(dim2):
            for k in xrange(dim3):
                old_idx[a_idx] = i
                old_idx[b_idx] = j
                old_idx[c_idx] = k
                old_val = tensor[old_idx[0], (n_col * old_idx[2]) + old_idx[1]]
                # new index
                permuted_tensor[i, (dim2 * k) + j] = old_val

    return permuted_tensor


def khatri_rao_prod(a, b):
    """Khatri-Rao product
    
    Generate Khatri-Rao product from 2 2-D matrix.

    Parameters
    ----------
    a : 2D array, shape (n, k)
        first matrix

    b : 2D array, shape (m, k)
        second matrix

    Returns
    -------
    matrix : 2D array, shape (n * m, k)
        Khatri-Rao product of `a` and `b`
    """

    a_row, a_col = a.shape
    b_row, b_col = b.shape
    # check column size
    if a_col != b_col:
        raise ValueError("column dimension mismatch: %d != %d" %
                         a_col, b_col)
    matrix = np.empty((a_row * b_row, a_col))
    for i in xrange(a_col):
        matrix[:, i] = np.kron(a[:, i], b[:, i])
    return matrix


def tensor_3d_prod(tensor, a, b, c):
    """Calculate product of 3D tensor with matrix on each dimension
        
    TODO: move it to test

    Parameters
    ----------
    tensor : 3D array, shape (n1, n2, n3)
    a : array, (n1, m)

    b :  array, (n2, n)

    c :  array, (n3, p)

    Returns
    -------
    t_abc : array, (m, n, p)
        tensor(a, b, c)

    """
    n1, n2, n3 = tensor.shape
    n1_, m = a.shape
    n2_, n = b.shape
    n3_, p = c.shape

    # (n1, n2, p)
    t_c = np.dot(tensor, c)

    t_bc = np.empty((n1, n, p))
    for i in xrange(n1):
        # (n, p) = (n, n2) * (n2, p)
        t_bc[i, :, :] = np.dot(b.T, t_c[i, :, :])

    t_abc = np.empty((m, n, p))
    for i in xrange(p):
        t_abc[:, :, i] = np.dot(a.T, t_bc[:, :, i])
    return t_abc
