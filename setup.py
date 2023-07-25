# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name="fastsam",
    version="0.1.0",
    install_requires=[],
    url="https://github.com/CASIA-IVA-Lab/FastSAM",
    packages=find_packages(exclude=["assets"])
)