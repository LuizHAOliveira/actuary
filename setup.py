from setuptools import setup
  
setup(
    name='actuary',
    version='0.1.0',
    description='A python package for calculating actuarial reserves.',
    author='Luiz Oliveira',
    author_email='lui.zhen14@hotmail.com',
    packages=['actuary', 'tests'],
    install_requires=[
        'numpy',
        'pandas',
    ],
)