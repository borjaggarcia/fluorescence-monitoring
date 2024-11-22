#!/usr/bin/env python

from setuptools import setup, find_packages

import rgbdriverkit

with open('README') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

# pypi classifiers
classifiers = [
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries :: Python Modules',
        ]

setup(
    name='pyrgbdriverkit',
    version=rgbdriverkit.__version__,
    description='Python Driver for RGB Photonics Devices',
    long_description=readme,
    author='RGB Photonics GmbH',
    author_email='support@rgb-photonics.com',
    url='http://rgb-photonics.com',
    platforms=["Linux"],
    install_requires=['pyusb>=1.0.0'],
    classifiers=classifiers,
    license=license,
    #packages=find_packages(exclude=('tests', 'docs'))
    packages=['rgbdriverkit'],
    package_dir={'rgbdriverkit': 'rgbdriverkit'},
    data_files=[('', ['etc/51-rgbdevices.rules'])],
)
