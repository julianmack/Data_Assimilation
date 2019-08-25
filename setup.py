from setuptools import setup, find_packages
from glob import glob

setup(
    name='VarDACAE',
    version='0.2',
    description='Pipeline for CAE-based Variational Data Assimilation',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    install_requires = ["pyevtk",
    "torch",
    "torchvision",
    "vtk",],
    python_requires='>=3',
    long_description=open('README.md').read(),
)