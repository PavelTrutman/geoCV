#!/usr/bin/python3

import unittest
import numpy as np
import geoCV

class TestLinalg(unittest.TestCase):
  """
  Unit test for linalg.py.

  by Pavel Trutman, pavel.trutman@cvut.cz
  """


  def testNullRandom(self):
    """
    Test null space of random matrices.
    """

    dims = [5, 10, 15, 25, 50, 100]
    for i in range(len(dims)):
      with self.subTest(i = i):
        A = np.random.rand(dims[i], dims[i])
        U, s, V = np.linalg.svd(A, full_matrices=True)
        r = 0
        for j in range(len(s)):
          if np.random.random_sample() > 0.7:
            s[j] = 0
          else:
            r += 1
        As = U.dot(np.diag(s)).dot(V)

        null = geoCV.linalg.null(As)
        self.assertLessEqual(np.linalg.norm(As.dot(null.dot(np.random.rand(dims[i] - r, 1)))), 1e-3)


  def testRandomRotation(self):
    """
    Test random rotation matrices generation.
    """

    for i in range(50):
      with self.subTest(i = i):
        R = geoCV.linalg.randomRotation()

        self.assertLessEqual(np.linalg.norm(R.dot(R.T) - np.eye(3)), 1e-3)


if __name__ == '__main__':
  unittest.main()
