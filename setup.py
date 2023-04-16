from setuptools import setup

with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

setup(
    name="marc_thesis",
    version="1.0",
    description="Code for my master's thesis",
    url="https://github.com/marc-gav/MasterThesis",
    author="Marc Gavilan Gil",
    author_email="marcgavilangil@gmail.com",
    license="MIT",
    install_requires=required_packages,
    # files are in marc_thesis folder
    packages=["marc_thesis"],
)
