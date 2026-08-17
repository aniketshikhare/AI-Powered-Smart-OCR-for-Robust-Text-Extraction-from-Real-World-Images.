"""
Setup script for AI-Powered Smart OCR
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-powered-smart-ocr",
    version="1.0.0",
    author="Aniket Shikhare",
    author_email="aniket@example.com",
    description="AI-Powered Smart OCR for Robust Text Extraction from Real-World Images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aniketshikhare/AI-Powered-Smart-OCR-for-Robust-Text-Extraction-from-Real-World-Images",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "smart-ocr=src.cli:main",
        ],
    },
)