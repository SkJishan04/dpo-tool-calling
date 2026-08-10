from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dpo-tool-calling",
    version="0.1.0",
    author="Your Name",
    description="DPO for Tool/Function Calling - Fine-tune models to intelligently use APIs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "trl>=0.7.4",
        "pydantic>=2.5.0",
        "pyyaml>=6.0.1",
        "datasets>=2.14.5",
        "numpy>=1.24.3",
        "pandas>=2.1.1",
        "scikit-learn>=1.3.2",
        "tqdm>=4.66.1",
        "wandb>=0.15.12",
    ],
)