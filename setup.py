# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]
setup(
    name="fastsam",
    version="0.1.1",
    install_requires=REQUIREMENTS,
    packages=["fastsam", "fastsam_tools"],
    package_dir= {
        "fastsam": "fastsam",
        "fastsam_tools": "utils",
    },
    url="https://github.com/CASIA-IVA-Lab/FastSAM",
)
