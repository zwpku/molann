import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepcv",
    version="0.0.6",
    author="Wei Zhang",
    author_email="wzhangpku@gmail.com",
    description="Find collective variables of molecular systems by training deep neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zwpku/deepcv",
    project_urls={
#        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
    ],
    package_dir={"": "deepcv"},
    packages=setuptools.find_packages(where="deepcv"),
    python_requires=">=3.6",
)
