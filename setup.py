#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8

import setuptools
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

if __name__ == '__main__':
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
        packages=setuptools.find_packages(),
        python_requires=">=3.7",
    )



