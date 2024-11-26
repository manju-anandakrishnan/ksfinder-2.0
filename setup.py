from setuptools import setup, find_packages
# List of requirements
requirements = []  # This could be retrieved from requirements.txt
# Package (minimal) configuration
setup(
    name="ksf2",
    version="1.0.0",
    description="KSFinder 2.0 dependent modules",
    packages=find_packages(),  # __init__.py folders search
    #install_requires=requirements
)