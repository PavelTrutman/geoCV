#!/usr/bin/python3

from .coordinates import Coordinates
import numpy as np

class Linalg:
  """
  Linear algebra functions.

  by Pavel Trutman, pavel.trutman@cvut.cz
  """

  def null(A, atol=1e-13, rtol=0):
    """
    Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value decomposition of 'A'.

    If both 'atol' and 'rtol' are positive, the combined tolerance is the maximum of the two; that is:
      tol = max(atol, rtol * smax)
    Singular values smaller than 'tol' are considered to be zero.

    Args:
      A (array): a 2D matrix
      atol (float): the absolute tolerance for a zero singular value. Singular values smaller than 'atol' are considered to be zero.
      rtol (float): the relative tolerance. Singular values less than rtol*smax are considered to be zero, where smax is the largest singular value.


    Returns:
      array: if 'A' is an array with shape (m, k), then it retuns an array with shape (k, n), where n is the estimated dimension of the nullspace of 'A'. The columns are a basis for the nullspace.
    """

    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


  def randomRotation():
    """
    Generates random rotation matrix of size 3x3.

    Firstly, a quaternion from normal distribution is sampled and then is is converted to the matrix representation.

    Args:

    Returns:
      array: 3x3 rotation matrix
    """

    q = np.random.randn(4)
    q = Coordinates.normalize(q)
    R = np.array([[q[0, 0]**2 + q[1, 0]**2 - q[2, 0]**2 - q[3, 0]**2, 2*q[1, 0]*q[2, 0] - 2*q[0, 0]*q[3, 0], 2*q[1, 0]*q[3, 0] + 2*q[0, 0]*q[2, 0]], [2*q[1, 0]*q[2, 0] + 2*q[0, 0]*q[3, 0], q[0, 0]**2 - q[1, 0]**2 + q[2, 0]**2 - q[3, 0]**2, 2*q[2, 0]*q[3, 0] - 2*q[0, 0]*q[1, 0]], [2*q[1, 0]*q[3, 0] - 2*q[0, 0]*q[2, 0], 2*q[2, 0]*q[3, 0] + 2*q[0, 0]*q[1, 0], q[0, 0]**2 - q[1, 0]**2 - q[2, 0]**2 + q[3, 0]**2]])

    return R
