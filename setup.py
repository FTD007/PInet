# install using 'pip install -e .'

from setuptools import setup

setup(name='pinet',
      packages=['pinet'],
      package_dir={'pinet': 'pinet'},
      install_requires=['torch',
                        'tqdm',
                        'plyfile'],
    version='0.0.1')
