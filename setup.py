from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sarsampling-iec62209",
    version="0.1.0",
    author="Cedric Bujard",
    author_email="bujard@itis.swiss",
    description="A Latin Hypercube Sampler for SAR (Specific Absorption Rate) testing scenarios",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cbujard/sarsampling-iec62209",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
