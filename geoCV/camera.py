#!/usr/bin/python3

from .coordinates import Coordinates
import numpy as np

class Camera:
  """
  Helper functions for camera projection matrices.

  by Pavel Trutman, pavel.trutman@cvut.cz
  """

  def P2KRC(P):
    """
    Camera matrix decomposition to K, R, C, that holds P = K*R*[I  -C]

    Args:
      P (array): cmera projection matrix

    Returns:
      array, array, array: calibration matrix K, rotation matrix R, position vector C
    """

    P = Camera.normalizeP(P)

    A = P[0:3, 0:3]
    C = Coordinates.columnize(-np.linalg.pinv(A).dot(P[:, 3]))

    K, R = Camera.A2KR(A)

    return K, R, C


  def A2KR(A):
    """
    Matrix decomposition A = K*R, K - upper triangular matrix, R - rotation matrix.

    Args:
      A (array): matrix to decompose

    Returns:
      array, array: calibration matrix K, rotation matrix R
    """

    R = np.zeros((3, 3))

    R[2, :] = np.squeeze(Coordinates.normalize(A[2, :]))
    R[0, :] = np.cross(A[1, :], R[2, :])
    R[0, :] = np.squeeze(Coordinates.normalize(R[0, :]))
    R[1, :] = np.cross(R[2, :], R[0, :])
    K = A.dot(R.T)

    return K, R


  def normalizeP(P):
    """
    Remove scale from the camera projection matrix.

    Args:
      P (array): camera projection matrix

    Returns:
      array: normalized camera projection matrix
    """

    d = np.linalg.det(P[0:3, 0:3])
    if abs(d) > 0:
      P = P*np.sign(d)

    scale = np.linalg.norm(P[2, 0:3])
    P = P/scale
    return P
