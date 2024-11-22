#!/usr/bin/env python

__author__ = "RGB Photonics GmbH"

# FixMe: Use six for Python 2/3 compatibility library
# import abc, six
# @six.add_metaclass(abc.ABCMeta)
# class SomeAbstractClass():
#   @abc.abstractmethod

try:
    #from abc import ABC, abstractmethod, abstractproperty # Python 3.4+
    import abc
    ABC = abc.ABCMeta('ABC', (object,), {})
except:
    #from abc import ABCMeta, abstractmethod, abstractproperty # Python 2.7
    import abc
    ABC = abc.ABCMeta('ABC', (object,), {}) # Python 2 and 3 compatible

import logging

_logger = logging.getLogger('rgb.device')

# ToDo: Add Exceptions class from Exceptions

class Device(ABC):
    """Base class for all device drivers"""

    def __init__(self):
        try:
            _logger.debug("Instantiating " + str(__class__))
        except:
            _logger.debug("Instantiating " + str(Device.__class__))

        self._isopen = False
        self._manufacturer = ""
        self._model_name = ""
        self._serial_number = ""
        self._port_name = ""

    @abc.abstractmethod
    def open(self):
        """Opens the connection to the device."""
        pass

    @abc.abstractmethod
    def close(self):
        """Closes the connection to the device."""
        pass

    @staticmethod
    def search_devices():
        """Searches for devices of this kind"""
        raise NotImplementedError("Method needs to be defined by sub-class.")

    def check_device_removed(self):
        """Checks whether the device was removed."""
        return False

    @property
    def isopen(self):
        """Indicates whether the connection to the device is open."""
        return self._isopen

    @property
    def manufacturer(self):
        """Gets the device manufacturer."""
        return self._manufacturer

    @property
    def model_name(self):
        """Gets the device model name."""
        return self._model_name

    @property
    def serial_number(self):
        """Gets the serial number.
        May be empty, if serial number cannot be read from the device."""
        return self._serial_number

    @property
    def port_name(self):
        """Gets the port name.
        May be empty, if not applicable."""
        return self._port_name

    @property
    def detailed_device_name(self):
        """ Gets a detailed device name.
        Note to implementors: This string should be unique and correct after
        initialization and after device object is returned from search_devices()."""
        s = self._model_name
        if self._serial_number != "":
            s = s + " (s/n: " + self._serial_number + ")"
        if self._port_name != "":
            s = s + " on port " + self._port_name
        return s

    def issamedeviceas(self, other_device):
        if other_device is None:
            raise ValueError()
        return other_device.detailed_device_name == self.detailed_device_name

