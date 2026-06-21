from setuptools import setup, find_packages

setup(
    name="xcardamagenet",
    version="2.0.0",
    description="Hybrid CNN-Transformer architecture for vehicle damage detection",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "einops>=0.7.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
    ],
)
