from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="bertologist",
    version="1.0.0",
    author="Marc Gavilan Gil",
    description="Master Thesis: Description of BERT clusters using salience scores",
    install_requires=required,
    packages=find_packages(),
)
