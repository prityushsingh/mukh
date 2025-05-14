from setuptools import find_packages, setup


def read_requirements(file):
    with open(file, "r", encoding="utf-8") as f:
        return f.read().splitlines()


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mukh",
    version="0.0.2",
    author="Ishan Dutta",
    author_email="duttaishan098@gmail.com",
    description="A python package to perform a variety of tasks on face images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ishandutta0098/mukh",
    project_urls={
        "Bug Tracker": "https://github.com/ishandutta0098/mukh/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
            "bump2version>=1.0.0",
        ],
    },
)
