from setuptools import setup, find_packages

setup(
    name="textregress",
    version="1.2.0",
    description="A package for performing advanced regression on text data using unified deep learning framework.",
    author="Jinhang Jiang, Ben Liu, Weiyao Peng, Karthik Srinivasan",
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
        "huggingface-hub"
    ],
    python_requires=">=3.6",
)
