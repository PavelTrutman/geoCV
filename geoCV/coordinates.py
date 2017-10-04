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
    Converts coordinates of points from affine coordinates to homogenous coordinates.

    Args:
      points (array): coordinates of points as columns

    Returns:
      array: homogenous coordinates
    """

    return np.vstack((points, np.ones((1, points.shape[1]))))


  def h2a(points):
    """
    Converts coordinates of points from homogenous coordinates to affine coordinates.

    Args:
      points (array): coordinates of points as columns

    Returns:
      array: affine coordinates
    """

    return (points/np.matlib.repmat(points[-1, :], points.shape[0], 1))[:-1, :]


  def calibrate(points, K):
    """
    Calibrates points by the given calibration matrix.

    Args:
      points (array): coordinates of points in columns
      K (array): calibration matrix

    Returns:
      array: calibrated points
    """

    Kinv = np.linalg.inv(K)
    return Kinv.dot(points)
