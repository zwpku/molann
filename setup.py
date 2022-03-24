import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="molcv",
    version="0.0.9",
    author="Wei Zhang",
    author_email="wzhangpku@gmail.com",
    description="Find collective variables of molecular systems by training deep neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zwpku/molcv",
    project_urls={
#        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
