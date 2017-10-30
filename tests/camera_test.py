#!/usr/bin/python3

import unittest
import numpy as np
import geoCV

class TestCamera(unittest.TestCase):
  """
  Unit test for camera.py.

  by Pavel Trutman, pavel.trutman@cvut.cz
  """

  def testA2KR(self):
    """
    Test A to K and R decomposition.
    """

    for i in range(50):
      with self.subTest(i = i):
        A = np.random.rand(3, 3)
        K, R = geoCV.camera.A2KR(A)

        self.assertLessEqual(np.linalg.norm(A - K.dot(R)), 1e-3)


  def testP2KRC(self):
    """
    Test projection camera matrix to K, R, C decomposition.
    """

    for i in range(50):
      with self.subTest(i = i):
        R = geoCV.linalg.randomRotation()
        C = np.random.rand(3, 1)
        K = np.random.rand(3, 3)
        K[1, 0] = 0
        K[2, 0] = 0
        K[2, 1] = 0
        K[2, 2] = 1
        P = K.dot(np.hstack((R, -R.dot(C))))
        Ke, Re, Ce = geoCV.camera.P2KRC(P)

        self.assertLessEqual(np.linalg.norm(K - Ke), 1e-3)
        self.assertLessEqual(np.linalg.norm(R - Re), 1e-3)
        self.assertLessEqual(np.linalg.norm(C - Ce), 1e-3)


  def testNormalizeP(self):
    """
    Test normalization of the camera projection matrix.
    """

    for i in range(50):
      with self.subTest(i = i):
        P = np.random.rand(3, 4)
        n = np.linalg.norm(geoCV.camera.normalizeP(P)[2, 0:3])
        self.assertLessEqual(n - 1, 1e-3)


if __name__ == '__main__':
  unittest.main()
