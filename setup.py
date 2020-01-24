#!/usr/bin/env python

from distutils.core import setup
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='seduce_ml',
      version='1.0',
      description='Experiment with SeDuCe',
      author='SeDuCe Cloud',
      author_email='seducecloud@gmail.com',
      url='https://github.com/SeduceProject/seduce_ml',
      package_dir={'': '.'},  # Optional
      packages=["seduce_ml"] + ["seduce_ml/"+p for p in find_packages(where="seduce_ml")],  # Required
      install_requires=requirements,
      )
