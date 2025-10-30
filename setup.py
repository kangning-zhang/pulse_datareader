from setuptools import setup, find_packages

setup(
    name='pulsepytools',
    version='0.1.0',
    # packages=['pulsepytools'],
    packages=find_packages(),
    description="Collected tools for annotating and using the PULSE data",
    include_package_data=True,
)
