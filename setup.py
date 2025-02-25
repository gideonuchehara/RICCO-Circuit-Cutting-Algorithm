from setuptools import setup, find_packages

# Read the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="RICCO-Circuit-Cutting-Algorithm",  # Name of package
    version="0.0.1",  # Initial version
    author="Gideon Uchehara", 
    author_email="gideonuchehara@gmail.com", 
    description="A framework for circuit cutting.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gideonuchehara/RICCO-Circuit-Cutting-Algorithm.git", 
    packages=find_packages(include=['ricco', 'utils']),
    # package_dir={"": "ricco"},  # Maps root package to `ricco` directory
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.0.0", 
    install_requires=[
        "pennylane==0.28.0", 
        "numpy==1.23.5",
    ],
    extras_require={
        "dev": ["pytest", "opt_einsum", "jupyter"],  # Additional packages for development
    },
    include_package_data=True,
)
