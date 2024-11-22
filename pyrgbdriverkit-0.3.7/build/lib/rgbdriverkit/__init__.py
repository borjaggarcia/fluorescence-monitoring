import logging

"""
RgbDriverKit Python API

This package exports the following modules and subpackages:

    spectrometer - the base class for spectrometers
    calibratedspectrometer - a spectrometer that supports pre-processing of the spectra
    qseriesdriver - a device class for controlling new Qseries spectrometers
"""

__author__ = 'RGB Photonics GmbH'

version_info = (0, 3, 7)
__version__ = '%d.%d.%d' % version_info

__all__ = ['qseriesdriver', 'calibratedspectrometer', 'spectrometer']

