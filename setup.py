from setuptools import setup

setup(
    name="timescales",
    version="0.0.1",
    description="Multi-timescale adaptive whitening with neural networks",
    author="Lyndon Duong",
    license="MIT",
    packages=["timescales"],
    url="https://github.com/lyndond/multi_timescale_whitening",
    install_requires=[
        "numpy",
        "seaborn",
        "scipy",
        "tqdm",
        "scikit-learn",
        "jupyter",
    ],
)
