#!/usr/bin/env python

__author__ = "RGB Photonics GmbH"

from datetime import datetime

import rgbdriverkit.spectrometer as spectrometer
from rgbdriverkit.helpers import enum
from rgbdriverkit.spectrometer import SpectrometerStatus

import logging

_logger = logging.getLogger('rgb.calibrspectr')

# Try to load module and store class name to separate variable
try:
    import rgbdriverkit.calibrspectrometerdata as calibratedspectrometerdata
    _calibratedspectrometerdata_class = calibratedspectrometerdata.CalibratedSpectrometer
    _USE_INTERNAL = 1 # Use module calibratedspectrometerdata for internal use
except ImportError:
    _calibratedspectrometerdata_class = object
    _USE_INTERNAL = 0

class CalibratedSpectrometer(spectrometer.Spectrometer, _calibratedspectrometerdata_class):
    """A spectrometer that supports pre-processing of the spectra.

    Please note that the processing steps are now performed entirely on-board.
    'Pre-Processing' means correcting a spectrum after reading it from the device
    and before handing them over to the calling program. These processing steps
    are applied in order to provide a more precise and meanigful spectrum.
    They are not application-specific.

    The different processing steps are listet in the SpectrometerProcessing enumeration.
    """

    def __init__(self):

        try: # Python 3
            _logger.debug("Instantiating " + str(__class__))
            super().__init__()
        except: # Python 2
            _logger.debug("Instantiating " + str(CalibratedSpectrometer.__class__))
            super(CalibratedSpectrometer, self).__init__() # Python 2 equivalent for new-style classes

        self.dark_set_values(None, None, None, 0) # clear dark spectra
        self._time_stamp = datetime
        self._load_level = None
        self._available_processing_steps = None
        self._processing_steps = None
        self._default_processing_steps = None
        self._previous_processing = 0
        # Processing step: Adjust offset
        # IMPORTANT: The following fields may be accessed from an exposure thread. They must therefore not be changed after initialization.
        self._previousOffsetAvg = -1e30
        self._previousDarkAvg = -1e30
        self._firstDarkPixel = 0
        self._numDarkPixels = 0
        self._firstOffsetPixel = 0
        self._numOffsetPixels = 0
        self._firstRealPixel = 0
        self._numDataValues = 0 # The number of pixels in an unprocessed raw spectrum (including dark and dummy pixels).
        self._readoutNoise = 20 # for a single exposure. Change this during initialization if better value is known.
        self._mirrorSpectrum = False
        # Processing step: Correct nonlinearity
        self._nonlinearity_coefficients = [1.0, 0.0, 0.0, 0.0]
        self._alternating_nonlinearity = False
        # Processing step: Remove permanent bad pixels
        # Processing step: Subtract dark
        self.__dark_exposure_time = -1.0
        self._dark_spec_times = [] # None?
        self.__dark_pixel_avg = None # None if not applicable
        self.__dark_spec = None # None? dark_spec[exp_time_index, pixel]
        self._dark_spec_type = 0
        self._currentDarkSpec = [] # None?
        self._currentDarkPixelAvg = float
        # Processing step: Remove temporary bad pixels
        # Processing steps: Normalize exposure time
        #                   Sensitivity calibration
        #                   Sensitivity smoothing
        #                   ScaleTo16BitRange
        self._sensitivity_calibration = None
        self.__sensitivity_unit = SpectrometerUnits.Unknown
        self.__sensitivity_description = ""
        self._sensitivity_smoothing_width = 0
        self._sensitivity_smoothing = None # warning: inverse values as compared to sensitivity_calibration
        # Processing step: Additional Filtering

        # Calibration
        self._calibration_load_error = ""
        self._can_restore_factory_calibration = False

        self._calibration_temperature_wavelengths = float('nan')
        self._calibration_temperature_nonlinearity = float('nan')
        self._calibration_temperature_dark_spectra = float('nan')
        self._calibration_temperature_sensitivity = float('nan')

        self._current_spectrum = None # means exposure not startedd or spectrum alreadout read out
        # current_spectrum.Data == None means exposure started, but not finished yet

    @property
    def model_id(self):
        """Gets the model ID (or an empty string, it not available).
        For devices made by RGB Photonics, this value starts with "ID" and is followed by a 4-digit number.
        For other devices this may be any kind of string."""
        return ""

    @property
    def hardware_version(self):
        """Gets the device hardware version."""
        return ""

    @property
    def software_version(self):
        """Gets the device firmware version."""
        return ""

    @property
    def time_stamp(self):
        return self._time_stamp

    @property
    def load_level(self):
        return self._load_level

    # --- Selection of processing steps

    @property
    def available_processing_steps(self):
        """Gets the processing steps that are available for this spectrometer."""
        return self._available_processing_steps

    @property
    def processing_steps(self):
        """Gets or sets the processing steps that are currently used when taking a spectrum."""
        return self._processing_steps
    @processing_steps.setter
    def processing_steps(self, value):
        self.processing_steps = value & self._available_processing_steps

    @property
    def default_processing_steps(self):
        """Gets the default processing steps."""
        return self._default_processing_steps

    @property
    def raw_data(self):
        """Gets or sets a value indicating whether the raw data from the image sensor should be returned."""
        return (self._processing_steps == 0)
    @raw_data.setter
    def raw_data(self, value):
        if (value == (self._processing_steps == 0)):
            return # if no change in raw_data: nothing to do here
        if value: # if raw_data changes from False to True
            self._previous_processing = self._processing_steps # remember previous processing steps
            self._processing_steps = 0 # and set raw_data to True
        else: # if raw_data changes from True to False
            if (self._previous_processing != 0): # if the raw_data=True state was set here (otherwise: not sure what to do)
                self._processing_steps = self._previous_processing # restore previous state
            self._previous_processing = 0 # don't use this value next time

    # --- Processing spectrum

    def get_spectrum(self):
        """If you need to override it in a descendent class, please override get_spectrum_data() instead."""
        spd = self.get_spectrum_data()
        return spd.Spectrum

    def get_spectrum_data(self):
        """Gets the spectrum including some metadata.
        Returns an instance of the SpectrumData class that contains the data."""
        if ( self._current_spectrum == None or self._current_spectrum.Spectrum == None ):
            raise IOError("No data to read")
        # (alternatively one could test for available_spectra == 0)
        # AdjustOffset should have been done during readout
        # ToDo: processing
        spd = self._current_spectrum
        self._current_spectrum = None
        self._load_level = spd.LoadLevel
        self._time_stamp = spd.TimeStamp
        return spd

    @property
    def nonlinearity_coefficients(self):
        """Gets or sets the nonlinearity coefficients."""
        return self._nonlinearity_coefficients
    @nonlinearity_coefficients.setter
    def nonlinearity_coefficients(self, value):
        self._nonlinearity_coefficients = value
        if self.can_read_temperature:
            #self._calibration_temperature_nonlinearity = self.Temperature
            pass

    def dark_set_values(self, dark_spec_times, dark_pixel_avg, dark_spec, dark_spec_type):
        pass

    # --- User Data

    def load_user_data(self):
        """Loads user data from the device.
        If supported by the spectrometer, you can use the User Data memory to store own data on the device."""
        return None

    def save_user_data(self):
        """Save user data to the device.
        If supported by the spectrometer, you can use the User Data memory to store your own data on the device"""
        raise NotImplementedError("This spectrometer cannot store user data.")

    # --- Calibration Loading, Saving and Temperature

    @property
    def calibration_load_error(self):
        """Gets a string that indicates if an error occured while loading the calibration data.
        If an error occurs while loading the calibration data, the default calibration is loaded and
        no exception thrown. You can read this property after loading a calibration or calling the
        Device.open() method in order to check if the calibration was loaded successfully."""
        return self._calibration_load_error

    def save_user_calibration(self):
        """Saves the user calibration.
        The implementation saves the calibration data to the hard disk.
        Override this method in order to save calibration data to the device."""
        raise NotImplementedError("Method not implemented yet.")

    def save_user_calibration_to_file(self, filename):
        """Saves the user calibration to a file."""
        raise NotImplementedError("Method not implemented yet.")

    def load_user_calibration(self):
        """Loads the user calibration.
        This implementation loads the calibration data from the hard disk.
        Override this method in order to load calibration data from the device."""
        if self.status != SpectrometerStatus.Idle:
            raise IOError("Spectrometer is not idle.")
        # if file exists: load from file
        # else:
        self._load_default_calibration("Calibration data file not found.")

    def load_user_calibration_from_file(self, filename):
        raise NotImplementedError("Method not implemented yet.")

    @property
    def calibration_data_path(self):
        raise NotImplementedError("Method not implemented yet.")

    @property
    def can_restore_factory_calibration(self):
        """Gets a value indicating whether this device contains a factory calibration that can be restored."""
        return self._can_restore_factory_calibration

    def restore_factory_calibration(self, restore_wavelengths, restore_nonlinearity, restore_dark_spectra, restore_spectral_sensitivity):
        """Restores the factory calibration."""
        raise NotImplementedError("No factory calibration available.")

    @property
    def calibration_temperature_wavelengths(self):
        """Gets the device temperature at which the wavelengths were calibrated (in degree Celcius)."""
        return self._calibration_temperature_wavelengths

    @property
    def calibration_temperature_nonlinearity(self):
        """Gets the device temperature at which the nonlinearity was calibrated (in degree Celcius)."""
        return self._calibration_temperature_nonlinearity

    @property
    def calibration_temperature_dark_spectra(self):
        """Gets or sets the device temperature at which the dark spectra were calibrated (in degree Celcius)."""
        return self._calibration_temperature_dark_spectra

    @property
    def calibration_temperature_sensitivity(self):
        """Gets the device temperature at which the spectral sensitivity and power was calibrated (in degree Celcius)."""
        return self._calibration_temperature_sensitivity

    def check_temperature(self):
        """Checks if the stored calibrations are still valid at the current spectrometer.temperature."""
        raise NotImplementedError("Not supported.")

    @property
    def wavelength_coefficients(self):
        return self._wavelength_coefficients
    @wavelength_coefficients.setter
    def wavelength_coefficients(self, value):
        self._wavelength_coefficients = value

    # --- Autoexposure

    # --- Auxiliary Interface

    @property
    def aux_interface(self):
        """Gets the auxiliary interface.
        Some devices may have an auxiliary interface to control other peripherals.
        This property refers to an instance of a class that is derived from CommunicationInterface
        and used to send and receive data over this interface.
        This interface gets opened and closed automatically together with this device."""
        return None


class SpectrumData(object):
    """A class that stores a spectrum together with some metadata.
    This is the return value of the CalibratedSpectrometer.GetSpectrumData() method."""

    def __init__(self):
        try:
            _logger.debug("Instantiating " + str(__class__))
        except:
            _logger.debug("Instantiating " + str(SpectrumData.__class__))

        self.Spectrum = [] # The spectrum as a float array
        self.ExposureTime = float
        self.Averaging = 1
        self.TimeStamp = datetime
        self.LoadLevel = -1.0
        self.Temperature = -300.0
        self.AppliedProcessingSteps = 0
        self.IntensityUnit = SpectrometerUnits.Unknown
        self.SaturationValue = -1.0
        self.OffsetAvg = 0.0
        self.DarkAvg = 0.0
        self.ReadoutNoise = -1.0

# Enumerations

"""An enumeration representing units of the y axis of a spectrum delivered from a spectrometer."""
SpectrometerUnits = enum(
        Unknown = 0,
        ADCvalues = 1,
        ADCnormalized = 2,
        nWnm = 3,
        nWm2nm = 4,
        Wsrm2nm = 5,
        Wsrnm = 6,
        )

"""An enumeration representing the different pre-processing steps for spectra."""
SpectrometerProcessing = enum(
        AdjustOffset = 1,
        CorrectNonlinearity = 2,
        RemovePermanentBadPixels = 4,
        SubractDark = 8,
        RemoveTemporaryBadPixels = 16,
        CompensateStrayLight = 32,
        NormalizeExposureTime = 64,
        SensitivityCalibration = 128,
        SensitivitySmoothing = 256,
        AdditionalFiltering = 512,
        ScaleTo16BitRange = 1024,
        )

"""An enumeration representing the possible results of an auto-exposure control cycle."""
AutoExposureResults = enum(
        NotChanged = 0,
        Changed = 1,
        TooMuchLight = 2,
        TooLittleLight = 3,
        NotAvailable = 4, # deprecated
        )

"""An enumeration representing the different types of auxiliary interfaces."""
AuxInterfaces = enum(
        NonePresent = 0,
        I2C = 1,
        SPI = 2,
        RS232 = 3,
        I2CLowLevel = 101,
        )


