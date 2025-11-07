from setuptools import find_packages
from setuptools import setup

setup(
    name='dofbot_info',
    version='0.0.0',
    packages=find_packages(
        include=('dofbot_info', 'dofbot_info.*')),
)
