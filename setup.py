import setuptools

long_description = """This repository/package contains pre-built ∂Lux models of the Toliman optical system, and 
pre-built parametrised ∂Lux source objects for Alpha Centauri. ∂Lux is an open-source differentiable optical modelling 
framework harnessing the structural isomorphism between optical systems and neural networks, giving forwards models of 
optical system as a parametric neural network. ∂Lux is built in Zodiax which is an open-source object-oriented Jax 
framework built as an extension of Equinox for scientific programming. The primary goal of the Toliman mission is to 
discover Earth-sized exoplanets orbiting in Alpha Centauri, the closest star system to our own. To achieve this, the 
mission will employ a novel telescope design that will be able to detect subtle changes in the positions of the Alpha 
Centauri binary pair. These changes are caused by the gravitational reflex motion induced by an Earth-sized companion, 
and this cutting-edge technology will enable scientists to identify exoplanets too small to be detected by conventional 
telescopes. Toliman utilises a binary phase diffraction pupil to grasp the expected microarcsecond-scale astrometric 
signal."""

# DEPENDENCIES
# 1. What are the required dependencies?
with open("requirements.txt") as f:
    install_requires = f.read().splitlines()
# 2. What dependencies required to run the unit tests? (i.e. `pytest --remote-data`)
# tests_require = ['pytest', 'pytest-cov', 'pytest-remotedata']


setuptools.setup(
    python_requires=">=3.10,<4.0",
    name="dLuxToliman",
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
    packages=["dLuxToliman"],
)
