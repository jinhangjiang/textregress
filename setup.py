from setuptools import setup, find_packages

setup(
    name="textregress",
    version="1.0.0",
    description="A package for performing linear regression on text data using configurable encoders and deep learning.",
    author="Jinhang Jiang, Weiyao Peng, Karthik Srinivasan",
    author_email="jinhang@asu.edu",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "tqdm",
        "torch",
        "pytorch-lightning",
        "sentence-transformers",
        "transformers",
    ],
    python_requires=">=3.6",
)
