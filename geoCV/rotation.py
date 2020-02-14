#!/usr/bin/python3

from .coordinates import Coordinates
import numpy

class Rotation:
  """
  Helper functions for rotations.

  by Pavel Trutman, pavel.trutman@cvut.cz
  """

  def x_(v):
    """
    Creates skew symmetric matrix out of vector.

    Args:
      v (array): column vector

    Returns:
      array: skew symmetric matrix
    """

    v = Coordinates.columnize(v)
    M = numpy.zeros((3,3, v.shape[1]))

    M[0, 1, :] = -v[2, :]
    M[0, 2, :] = v[1, :]
    M[1, 2, :] = -v[0, :]

    M[1, 0, :] = v[2, :]
    M[2, 0, :] = -v[1, :]
    M[2, 1, :] = v[0, :]

    return numpy.squeeze(M)


  def x_2(v):
    """
    Squared skew symmetric matrix out of vector.

    Args:
      v (array): column vector

    Returns:
      array: squared skew symmetric matrix
    """

    v = Coordinates.columnize(v)
    M = numpy.empty((3,3, v.shape[1]))

    M[0, 0, :] = -v[1, :]**2 -v[2, :]**2
    M[1, 1, :] = -v[0, :]**2 -v[2, :]**2
    M[2, 2, :] = -v[0, :]**2 -v[1, :]**2

    M[0, 1, :] = v[0, :]*v[1, :]
    M[1, 0, :] = v[0, :]*v[1, :]
    M[0, 2, :] = v[0, :]*v[2, :]
    M[2, 0, :] = v[0, :]*v[2, :]
    M[1, 2, :] = v[1, :]*v[2, :]
    M[2, 1, :] = v[1, :]*v[2, :]

    return numpy.squeeze(M)


  def angleAxis2Matrix(v, angle=None):
    """
    Computes rotation matrix from anle axis representation.

    Args:
      v (array): column vector representing the rotation axis (normalized to 1)
      angle (float): angle in radians

    Returns:
      array: rotation matrix
    """

    v = Coordinates.columnize(v)

    if angle is None:
      angle = Coordinates.vectorNorm(v)
      v = numpy.true_divide(v, angle, where=(angle != 0))
    else:
      v = Coordinates.normalize(v)

    if not isinstance(angle, numpy.ndarray):
      angle = numpy.array([angle])
    n = angle.shape[0]
    I = numpy.tile(numpy.eye(3)[:, :, None], (1, 1, n))
    Rx = Rotation.x_(v)
    Rxx = Rotation.x_2(v)
    if n == 1:
      Rx = Rx[:, :, None]
      Rxx = Rxx[:, :, None]
    s = numpy.sin(angle)[None, None, :]
    c = numpy.cos(angle)[None, None, :]
    R = I + s*Rx + (1 - c)*Rxx

    return numpy.squeeze(R)


  def quaternion2Matrix(q):
    """
    Rotation matrix from quaternion formula.

    Args:
      q (array): quaternion as a column vector

    Returns:
      array: 3x3 rotation matrix
    """

    q = Coordinates.normalize(q)
    R = numpy.array([[q[0, 0]**2 + q[1, 0]**2 - q[2, 0]**2 - q[3, 0]**2, 2*q[1, 0]*q[2, 0] - 2*q[0, 0]*q[3, 0], 2*q[1, 0]*q[3, 0] + 2*q[0, 0]*q[2, 0]], [2*q[1, 0]*q[2, 0] + 2*q[0, 0]*q[3, 0], q[0, 0]**2 - q[1, 0]**2 + q[2, 0]**2 - q[3, 0]**2, 2*q[2, 0]*q[3, 0] - 2*q[0, 0]*q[1, 0]], [2*q[1, 0]*q[3, 0] - 2*q[0, 0]*q[2, 0], 2*q[2, 0]*q[3, 0] + 2*q[0, 0]*q[1, 0], q[0, 0]**2 - q[1, 0]**2 - q[2, 0]**2 + q[3, 0]**2]])

    return R


  def matrix2Quaternion(R):
    """
    Quaternion from rotation matrix formula.

    Args:
      R (array): 3x3 rotation matrix

    Returns:
      array: quaternion as a column vector
    """

    tr = numpy.trace(R)
    q = numpy.array([[tr + 1], [R[2, 1] - R[1, 2]], [R[0, 2] - R[2, 0]], [R[1, 0] - R[0, 1]]])/(2*numpy.sqrt(tr + 1))
    return q


  def matrix2AngleAxis(R):
    """
    Angle axis representation from rotation matrix.

    Args:
      R (array): rotation matrix

    Returns:
      float: angle
      array: rotation axis
    """

    angle = numpy.arccos(numpy.clip(0.5*(numpy.trace(R) - 1), -1, 1))
    if angle != 0:
      v = 0.5*numpy.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])/numpy.sin(angle)
      v = Coordinates.columnize(v)
    else:
      v = numpy.full((3, 1), numpy.nan)

    return v, angle


  def randomRotationMatrix():
    """
    Generates random rotation matrix of size 3x3.

    Firstly, a quaternion from normal distribution is sampled and then is is converted to the matrix representation.

    Args:

    Returns:
      array: 3x3 rotation matrix
    """

    q = numpy.random.randn(4)
    q = Coordinates.normalize(q)
    R = Rotation.quaternion2Matrix(q)

    return R
