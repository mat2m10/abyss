"""
Import setuptools
"""
from setuptools import setup
from setuptools import find_packages
import os

"""
Reading the requirements
"""
with open('requirements.txt') as f:
    content = f.readlines()
    requirements = [x.strip() for x in content]

"""
Setup of the project
"""
setup(name = 'abyss',
      description= "a population stratification correction tool that uses deep learning techniques",
      install_requires = requirements,
      packages=find_packages(),
      author= "Matthieu de Hemptinne",
      )