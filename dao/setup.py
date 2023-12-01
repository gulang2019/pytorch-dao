from setuptools import setup, find_packages

setup(
    name = 'dao',
    version = '0.1.0',
    description='DAO intelligent memory management for Machine Learning',
    package_dir = {"": "python"},
    packages = find_packages(where="python", include=["dao", "dao.*"]),
)