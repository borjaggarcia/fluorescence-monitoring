#!/usr/bin/env python

__author__ = "RGB Photonics GmbH"

try:
    from abc import ABC, abstractmethod, abstractproperty # Python 3.4+
except:
    from abc import ABCMeta, abstractmethod, abstractproperty # Python 2.7


import rgbdriverkit.devicedriver as devicedriver
from rgbdriverkit.helpers import enum

import logging

_logger = logging.getLogger('rgb.spectrometer')

class Spectrometer(devicedriver.Device):
    """Base class for spectrometers"""

    def __init__(self):
        try:
            _logger.debug("Instantiating " + str(__class__))
            super().__init__()
        except:
            _logger.debug("Instantiating " + str(Spectrometer.__class__))
            super(Spectrometer, self).__init__() # Python 2 equivalent new-style classes
        self._averaging = 1
        self._pixel_count = 0 # IMPORTANT: This value may be accessed from an exposure thread. It must therefore not be changed after initialization.
        self._min_exposure_time = 0
        self._max_exposure_time = 0
        self._max_averaging = 1
        self._wavelength_coefficients = [1.0, 1.0, 0.0, 0.0]
        self._num_io_pins = 0
        self._external_trigger_rising_edge = True
        self._trigger_option_available = [True, False, True]

    def get_exposure_time(self):
        pass
    def set_exposure_time(self, value):
        pass
    exposure_time = abstractproperty(get_exposure_time, set_exposure_time)

    # on_exposure_time_changed event handler

    @property
    def averaging(self):
        return self._averaging
    @averaging.setter
    def averaging(self, value):
        if value > self._max_averaging:
            raise ValueError("Averaging is too large")
        if value < 1:
            raise ValueError("Averaging must be positive")
        self._averaging = value
        #onaveragingchanged()

    # on_averaging_changed() event handler

    @abstractmethod
    def start_exposure(self):
        """Starts the exposure."""
        pass

    @abstractmethod
    def cancel_exposure(self):
        """Cancels the exposure."""
        pass

    @abstractproperty
    def status(self):
        """Gets the spectrometer status.
        One of the values of the SpectrometerStatus enumeration."""
        pass

    @abstractproperty
    def available_spectra(self):
        """Gets the number of spectra that are available, but have not been read out yet."""
        pass

    @abstractmethod
    def get_spectrum(self):
        """Gets the spectrum as a float array.
        The spectrum contains pixel_count values"""
        pass

    @property
    def pixel_count(self):
        """Gets the number of pixels in a spectrum.
        This is the number of values returned by get_spectrum(). It might differ from
        the internal number of data_values received from the device."""
        return self._pixel_count

    @property
    def min_exposure_time(self):
        """Gets the minimum exposure time in seconds."""
        return self._min_exposure_time

    @property
    def max_exposure_time(self):
        """Gets the maximum exposure time in seconds."""
        return self._max_exposure_time

    @property
    def max_averaging(self):
        """Gets the maximum averaging."""
        return self._max_averaging

    @property
    def wavelength_coefficients(self):
        """Gets the wavelength coefficients.
        Should be 4 elements containing constant [0], linear [1], quadratic [2] and cubic [3]
        term of the 3rd order polynominal used for calculating the wavelengths. Wavelengths are in nm."""
        return self._wavelength_coefficients
    @wavelength_coefficients.setter
    def wavelength_coefficients(self, value):
        self._wavelength_coefficients = value

    def get_wavelengths(self):
        _logger.debug("Compute wavelengths (in nm) from coefficients...")
        if (self._wavelength_coefficients == None or len(self._wavelength_coefficients) < 2):
            raise ValueError("Not enough wavelength coefficients")
        calibr0 = self._wavelength_coefficients[0]
        calibr1 = self._wavelength_coefficients[1]
        if len(self._wavelength_coefficients) > 2:
            calibr2 = self._wavelength_coefficients[2]
        if len(self._wavelength_coefficients) > 3:
            calibr3 = self._wavelength_coefficients[3]

        lambda_nm = [0] * self._pixel_count
        isqu = float
        for i in range(self._pixel_count):
            isqu = float(i * i)
            lambda_nm[i] = calibr3 * isqu * float(i) + calibr2 * isqu + calibr1 * float(i) + calibr0

        return lambda_nm

    @abstractproperty
    def time_stamp(self):
        """Gets the time stamp for the most recent spectrum.
        The date and time of the start of the exposure according to the system clock."""
        pass

    @abstractproperty
    def load_level(self):
        """Gets the sensor load level for the most recent spectrum.
        0 = no signal, 1.0 = maximum level for good signal, >= 1.0 = overload"""
        pass

    # --- Digital I/O Port

    @property
    def num_io_pins(self):
        return self._num_io_pins

    def set_io_pin_configuration(self, pin, config):
        """Configures an I/O pin with the pin number starting at 0."""
        raise ValueError("No I/O pins available on this device.")

    def get_io_pin_configuration(self, pin):
        """Gets the configuration for an I/O pin."""
        raise ValueError("No I/O pins available on this device.")

    def set_io_pin(self, pin, state):
        """Sets the IO pin output state with the pin number at 0."""
        if state:
            self.set_io_pin_configuration(pin, SpectrometerIOConfiguration.OutputConstantHigh)
        else:
            self.set_io_pin_configuration(pin, SpectrometerIOConfiguration.OutputConstantLow)

    def get_io_pin(self, pin):
        """Gets an IO pin input state with pin number starting at 0."""
        return ( (self.io_pins & (1 << pin)) != 0 )

    @property
    def io_pins(self):
        raise NotImplementedError("Not supported.")

    # --- Trigger

    @property
    def external_trigger_source(self):
        """Gets or sets the I/O pin to be used as the external trigger source."""
        return -1
    @external_trigger_source.setter
    def external_trigger_source(self, value):
        raise NotImplementedError("Not supported.")

    @property
    def external_trigger_rising_edge(self):
        """Gets or sets a value indicating whether to trigger on the rising or falling edge of the external trigger source."""
        return self._external_trigger_rising_edge
    @external_trigger_rising_edge.setter
    def external_trigger_rising_edge(self, value):
        self._external_trigger_rising_edge = value

    @property
    def trigger_option(self):
        """Gets or sets the trigger option."""
        return SpectrometerTriggerOptions.FreeRunningTriggerEnd
    @trigger_option.setter
    def trigger_option(self, value):
        raise NotImplementedError("Not supported.")

    def trigger_option_available(self, mode):
        """Gets whether a certain SpectrometerTriggerOptions is available."""
        return self._trigger_option_available[int(mode)]

    @property
    def use_external_trigger(self):
        """Gets or sets a value indicating whether to use the external trigger when taking spectra."""
        return False
    @use_external_trigger.setter
    def use_external_trigger(self, value):
        if value:
            raise NotImplementedError("Not supported.")

    @property
    def can_use_external_trigger(self):
        return False

    # --- Temperature sensor

    @property
    def can_read_temperature(self):
        """Gets a value indicating whether this device can measure its internal temperature."""
        return False

    @property
    def temperature(self):
        """Gets the internal device temperature. (in degree Celcius)"""
        raise NotImplementedError("Not supported.")

# Enumerations

"""An enumeration of the values returned by the Status property of a spectrometer class."""
SpectrometerStatus = enum(
        Idle = 0,
        WaitingForTrigger = 1,
        TakingSpectrum = 2,
        NotReady = -1,
        Busy = -2,
        Error = -3,
        Closed = -4,
        )

"""An enumeration representing the different trigger options for a spectrometer."""
SpectrometerTriggerOptions = enum(
        FreeRunningTriggerEnd = 0,
        FreeRunningTriggerStart = 1,
        HardwareTriggered = 2
        )

"""An enumeration of the spectrometer I/O configuraiton values."""
SpectrometerIOConfiguration = enum(
        OutputConstantLow = 0,
        OutputConstantHigh = 1,
        OutputDuringExpLow = 2,
        OutputDuringExpHigh = 3,
        Input = 4,
        OutputPulsed = 8,
        OutputDuringExpPulsedLow = 10,
        OutputDuringExpPulsedHigh = 11,
        )
