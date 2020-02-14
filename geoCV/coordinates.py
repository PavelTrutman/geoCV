#!/usr/bin/python3

import numpy as np
import numpy.matlib

class Coordinates:
  """
  Helper functions for coordinates.

  by Pavel Trutman, pavel.trutman@cvut.cz
  """

  def a2h(points):
    """
    Converts coordinates of points from affine coordinates to homogeneous coordinates.

    Args:
      points (array): coordinates of points as columns

    Returns:
      array: homogeneous coordinates
    """

    points = Coordinates.columnize(points)

    points = np.vstack((points, np.empty((1, points.shape[1]))))
    points[-1, :] = 1
    return points


  def h2a(points):
    """
    Converts coordinates of points from homogeneous coordinates to affine coordinates.

    Args:
      points (array): coordinates of points as columns

    Returns:
      array: affine coordinates
    """

    points = Coordinates.columnize(points)

    return Coordinates.normalize(points, homogeneous=True)[:-1, :]


  def columnize(vectors):
    """
    Check and fix if vector is a column.

    Args:
      vectors (array): array of vectors to verify

    Returns:
      array: coilumnized vectors if required
    """

    if len(vectors.shape) == 1:
      return Coordinates.onedim2col(vectors)
    else:
      return vectors


  def onedim2col(vector):
    """
    Converts one dimensional vector to a column vector.

    Args:
      vector (array): one dimensional vector

    Returns:
      array: 2D column vector
    """

    if len(vector.shape) != 1:
      raise ValueError('Vector must be one dimensional, has ' + str(len(vector.shape)) + ' dimensions.')

    return vector[:, np.newaxis]


  def calibrate(points, K):
    """
    Calibrates points by the given calibration matrix.

    Args:
      points (array): coordinates of points in columns
      K (array): calibration matrix

    Returns:
      array: calibrated points
    """

    points = Coordinates.columnize(points)

    Kinv = np.linalg.inv(K)
    return Kinv.dot(points)


  def undistort(points, radial):
    """
    Undistort points distorted by radial distorsion.

    Args:
      points (array): coordinates of points in columns
      radial (float): parameter of the radial distorsion model

    Returns:
      array: undistorted points
    """

    points = Coordinates.columnize(points)

    affine = Coordinates.h2a(points)
    radialCoefs = 1 + radial*numpy.sum(affine**2, axis=0)
    return Coordinates.a2h(affine*radialCoefs)

  def vectorNorm(vectors):
    """
    Computes columnvise 2 norm.

    Args:
      vectors (array): vectors in columns

    Returns:
      array: norm of columns
    """

    vectors = Coordinates.columnize(vectors)

    return np.linalg.norm(vectors, 2, axis=0)


  def normalize(vectors, homogeneous=False):
    """
    Normalize the given vectors.

    Args:
      vectors (array): vectors to normalize as columns
      homogeneous (bool): true if homogeneous coordinates given

    Returns:
      array: normalized vectors
    """

    vectors = Coordinates.columnize(vectors)

    if homogeneous:
      return vectors/np.matlib.repmat(vectors[-1, :], vectors.shape[0], 1)
    else:
      return vectors/np.matlib.repmat(Coordinates.vectorNorm(vectors), vectors.shape[0], 1)


  def angle(x, y):
    """
    Computes angles between vectors.

    Args:
      x (array): coordinates column wise
      y (array): coordinates column wise
    Returns:
      array: angles between the vectors
    """

    x = Coordinates.normalize(x)
    y = Coordinates.normalize(y)

    angleCos = numpy.abs(numpy.sum(x*y, axis=0))
    numpy.clip(angleCos, -1, 1, out=angleCos)
    angle = numpy.arccos(angleCos)
    return angle
