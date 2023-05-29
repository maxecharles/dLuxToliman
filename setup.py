import setuptools
import os
import codecs
import re

long_description = "A repo to hold the canonical dLux Toliman models."

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# DEPENDENCIES
# 1. What are the required dependencies?
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()
# 2. What dependencies required to run the unit tests? (i.e. `pytest --remote-data`)
# tests_require = ['pytest', 'pytest-cov', 'pytest-remotedata']


setuptools.setup(
    python_requires='>=3.7,<4.0',
    name="dLuxToliman",
    version=find_version("dLuxToliman", "__init__.py"),
    description=long_description,
    long_description=long_description,

    author="Max Charles",
    author_email="max.charles@sydney.edu.au",
    url="https://github.com/maxecharles/dLuxToliman",

    project_urls={
        "Bug Tracker": "https://github.com/maxecharles/dLuxToliman/issues",
    },

    install_requires=install_requires,

    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],

    packages=["dLuxToliman"]
)