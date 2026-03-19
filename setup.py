from setuptools import find_packages, setup

setup(
    name="wris",
    version="0.2.1",
    description="Wildfire Risk Intelligence System (WRIS) mini project",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
