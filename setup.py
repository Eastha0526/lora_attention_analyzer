from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="lora-attention-analyzer",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for analyzing attention maps in Stable Diffusion models with LoRA support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lora-attention-analyzer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Researchers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "isort",
        ],
        "optional": [
            "scikit-image",
            "scipy",
        ]
    },
    entry_points={
        "console_scripts": [
            "lora-attention-analyzer=lora_attention_analyzer.cli.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)