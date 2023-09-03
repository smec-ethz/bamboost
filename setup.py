from setuptools import setup, find_packages

setup(
    name='bamboost',
    version='0.3',
    description='A light weight database manager using HDF5',
    url='https://gitlab.ethz.ch/compmechmat/research/libs/dbmanager',
    author='florez',
    author_email='florez@ethz.ch',
    license='MIT',
    packages=find_packages(include=['bamboost', 'bamboost.*']),
    install_requires=[
        'pandas',
        'mpi4py',
        'numpy',
        ],
    # h5py is not in the install requirements because it would reinstall
    # messing with the installation for parallel support :/
)
