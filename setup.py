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
      package_dir={'': 'seduce_ml'},  # Optional
      packages=find_packages(where='package_name'),  # Required
      install_requires=requirements,
      )
