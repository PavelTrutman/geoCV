#!/usr/bin/python3

from setuptools import setup

"""
Setup script for Geometry for Computer Vision package.

by Pavel Trutman, pavel.tutman@cvut.cz
"""

setup (
  # Distribution meta-data
  name='geoCV',
  version='0.0.0',
  description='Package for maintaining geometry in computer vision.',
  long_description='Package for easy maintaining of geometry in computer vision.',
  author='Pavel Trutman',
  author_email='pavel.trutman@cvut.cz',
  url='https://github.com/PavelTrutman/GeoCV',
  package_dir={'GeoCV' : '.'},
  packages = ['geoCV'],
  test_suite='tests',
  install_requires=[
    'numpy',
  ],
)
