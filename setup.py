from setuptools import find_packages, setup
setup(
    name='Graph_EQA',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "imageio",
        "omegaconf",
    ],
    include_package_data=True,
)