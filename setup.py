from setuptools import setup, find_packages

setup(
    name="synthetic-review-generator",
    version="1.0.0",
    description="Production-grade synthetic review data generator with quality guardrails",
    author="AI Engineer",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if line.strip() and not line.startswith("#")
    ],
    entry_points={
        "console_scripts": [
            "generate-reviews=generate:main",
        ],
    },
    python_requires=">=3.9",
)