from setuptools import setup, find_packages

setup(
    name='dbmanager',
    version='0.1',
    description='A light weight database manager using HDF5',
    url='https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager',
    author='florez',
    author_email='florez@ethz.ch',
    license='MIT',
    packages=find_packages(include=['dbmanager', 'dbmanager.*']),
    install_requires=[
        'h5py',
        'pandas',
        'mpi4py',
        'numpy',
        ],
)
