from setuptools import setup, find_packages

setup(
    name='DApipeline',
    version='0.1',
    description='pipeline for AE-based Variational Data Assimilation',
    packages=find_packages(exclude=[]),
    long_description=open('README.md').read(),
)