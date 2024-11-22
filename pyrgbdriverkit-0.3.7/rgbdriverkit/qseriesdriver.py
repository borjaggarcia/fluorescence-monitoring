#!/usr/bin/env python

__author__ = "RGB Photonics GmbH"

import usb.core
import usb.util
import struct
import datetime as datetime

import rgbdriverkit.calibratedspectrometer as calibratedspectrometer
from rgbdriverkit.spectrometer import SpectrometerStatus
from rgbdriverkit.spectrometer import SpectrometerTriggerOptions

from rgbdriverkit.helpers import enum

import logging

_logger = logging.getLogger('rgb.qseries')

class Qseries(calibratedspectrometer.CalibratedSpectrometer):
    """Device class for controlling new Qseries spectrometers."""

    __ID_VENDOR = 0x276E
    __ID_PRODUCT_QMINI = 0x0208 # USB PID for Qmini devices (currently unused)
    __ID_PRODUCT_QRED = 0x0209 # USB PID for Qred devices (currently unused)
    __ID_PRODUCT_QWAVE = 0x020A # USB PID for Qwave devices (currently unused)

    __MIN_FIRMWARE_VERSION_REQUIRED = (2, 0, 6, 1) # Full support for processing steps
    __CALIBR_PAGE_SIZE = 4096

    def __init__(self, device_path):
        """Initializes a new instance of the Qseries class
        The device_path as returned from the search_devices() function.
        You can use this to access a spectrometer with a given device path."""

        try: # Python 3
            _logger.debug("Instantiating " + str(__class__))
            super().__init__()
        except:
            _logger.debug("Instantiating " + str(Qseries.__class__))
            super(Qseries, self).__init__() # Python 2 equivalent new-style classes

        self.__device_path = device_path # ToDo: Move to communication interface later
        self.__usbdev = device_path # use device path for instantiating the USB device
        self._model_name = self.__usbdev.product
        self._manufacturer = self.__usbdev.manufacturer
        self._serial_number = self.__usbdev.serial_number
        self._model = 661520904 # uint of vidpid (e.g. 0x276e0208 = 661520904)
        # FixMe: This is the default value that is updated when opening device connection.

        self._max_averaging = 1
        self._num_io_pins = 4

        self.__versioncode = None # version code used for legacy support
        self.__software_version = (0, 0, 0, 0)

        self.__exposure_time = 0
        # Calibration
        self.__calibr_num_pages = 0 # for each set of calibration values (factory and user)
        # user calibration starts at page 0, factory calibration starts at page calibr_num_pages
        self.__userdata_num_pages = 0

        self.__max_ccd_value = 2**16-1

        # Digital I/O ports
        self.__pin_config = [0] * 4
        self.__trigger_pin = 0
        self.__trigger_option = SpectrometerTriggerOptions.FreeRunningTriggerEnd
        self.__use_trigger = False

    def open(self):
        """Opens the connection to the device."""
        _logger.info("Open device connection...")

        self.close()
        self.usbOpen()
        self._isopen = True

        try:
            # Check Model ID
            try:
                model_id = self.__read_integer(_CommandCodes.GetDeviceID)
                _logger.debug("Model ID: 0x%08x", model_id)
                self._model = model_id
            except:
                _logger.error("Reading Model ID failed.")
                raise IOError("Device not found exception.")
            # ToDo: Add new exception: DeviceNotFoundException?

            # Get and check firmware version
            versioncode = self.__read_integer(_CommandCodes.GetSoftwareVersion)
            self.__software_version = ( ((versioncode>>24) & 255), ((versioncode>>16) & 255), ((versioncode>>8) & 255), (versioncode & 255 ) )
            if self.__software_version < (0, 1, 0, 0):
                raise ValueError("The device is waiting for a new firmware to be programmed.") # bootloader activated
            elif self.__software_version < Qseries.__MIN_FIRMWARE_VERSION_REQUIRED:
                raise ValueError("The device firmware version " + self.software_version + " is too old for this driver. Please update the firmware.")
            _logger.info("Reading device information data...")

            self._pixel_count = self.__read_integer(_CommandCodes.GetPixelCount)
            self._numDataValues = self.__read_integer(_CommandCodes.GetDataCount)
            self._firstOffsetPixel = self.__read_integer(_CommandCodes.GetFirstOffsetPixel)
            self._numOffsetPixels = self.__read_integer(_CommandCodes.GetNumOffsetPixels)
            self._firstDarkPixel = self.__read_integer(_CommandCodes.GetFirstDarkPixel)
            self._numDarkPixels = self.__read_integer(_CommandCodes.GetNumDarkPixels)
            self._firstRealPixel = self.__read_integer(_CommandCodes.GetFirstPixel)
            self._mirrorSpectrum = bool(self.__read_integer(_CommandCodes.GetMirrorSpectrum))
            self.__calibr_num_pages = self.__read_integer(_CommandCodes.GetCalibrationDataNumPages)
            self.__userdata_num_pages = self.__read_integer(_CommandCodes.GetUserDataNumPages)
            #self._sensitivitySmoothingWidth # FixMe: Not used in future versions?

            self._available_processing_steps = self.__read_integer(_CommandCodes.GetMaxProcessingSteps)
            self._default_processing_steps = self.__read_integer(_CommandCodes.GetDefaultProcessingSteps)
            self._processing_steps = self.__read_integer(_CommandCodes.GetProcessingSteps)
            self._min_exposure_time = self.__read_integer(_CommandCodes.GetMinExposureTime) * 1e-6
            self._max_exposure_time = self.__read_integer(_CommandCodes.GetMaxExposureTime) * 1e-6
            self._max_averaging = self.__read_integer(_CommandCodes.GetMaxAveraging)

            self.__exposure_time = self.__read_integer(_CommandCodes.GetExposureTime) * 1e-6
            self._averaging = self.__read_integer(_CommandCodes.GetAveraging)
            # if canset_targettemperature: targettemperature = read_integer()

            self.__max_ccd_value = self.__read_integer(_CommandCodes.GetMaxDataValue)
            self.__initialize_port() # initialize digital I/O ports
            self.__write_command(_CommandCodes.Reset)

            # Processing steps are performed entirely on-board.
            if calibratedspectrometer._USE_INTERNAL == 1: # Load only for internal use.
                _logger.info("Internal use: Loading of user calibration available")
                self.load_user_calibration()
            else:
                _logger.info("Using on-board processing")
                pass

            self._can_restore_factory_calibration = self.__factory_calibration_available
            # can_use_sensitivity_calibration
        except:
            self.close()
            raise

        _logger.info("Open device connection... Done")

    def close(self):
        if self._isopen:
            try:
                self._write_command(_CommandCodes.Bye) # allow device to save power
            except:
                pass
            self.usbClose()
            self._isopen = False
        _logger.info("Device connection closed.")

    @staticmethod
    def search_devices(serial_number=None):
        """Searches for spectrometers of this kind."""
        _logger.info("Searching for devices...")

        # Find device with unique serial number as custom match
        class find_serial_number(object):
            def __init__(self, serial_number_):
                self._serial_number = str(serial_number_) # can only compare str or unicode
            def __call__(self, device):
                _logger.debug("Apply custom match with serial number " + str(self._serial_number) + " on " + repr(device))
                # Check device if it contains the serial number
                if device.serial_number == self._serial_number:
                    return True
                return False

        # ToDo: Remove backend parameter: If no backend is supplied, PyUSB select one of the predefined backends
        #       according to system availability. Maybe use optional parameter to select backend.
        # Search only for devices that match our vendor ID
        if serial_number is None:
            devs_gen = usb.core.find(find_all=True, idVendor=Qseries.__ID_VENDOR)
        else:
            devs_gen = usb.core.find(find_all=True, idVendor=Qseries.__ID_VENDOR, custom_match=find_serial_number(serial_number))

        devs = list(devs_gen) # get list from returned generator object (even if no device was found)

        if len(devs) is 0:
            _logger.info("Could not find device.")
            return None

        # ToDo: How to determine if device is already in use?
        # FixMe: Disposing resources necessary? Probably not, because after find nothing changed state of USB device

        # Now we have found a device that matches, we can dispose it.
        for d in devs:
            _logger.debug("Disposing " + repr(d))
            usb.util.dispose_resources(d)

        return list(devs)

    @property
    def device_path(self):
        """Gets the USB device path for this spectrometer."""
        return self.__device_path

    #def issamedeviceas(self, other_device):

    # --- Device Support
    def device_reset(self):
        """Resets the device to the power-on state.
        The spectrometer connection is closed after the reset. You'll have to wait for
        the device to appear again and call open() afterwards."""
        _logger.debug("Reset the device to the power-on state.")
        if self.status != SpectrometerStatus.Idle:
            raise IOError("Spectrometer is not idle.")
        self.__write_command(_CommandCodes.Bye) # save parameters in the device flash
        self.__write_command(_CommandCodes.SystemReset)
        self.usbClose()
        self._isopen = False

    def parameter_reset(self):
        # also calls close(), because local variables in this class may not be valid anymore
        _logger.debug("Parameter reset.")
        if self.status != SpectrometerStatus.Idle:
            raise IOError("Spectrometer is not idle.")
        self.__write_command(_CommandCodes.ParameterReset)
        self.close()


    # --- Properties

    @property
    def hardware_version(self):
        v = self.__read_integer(_CommandCodes.GetHardwareVersion)
        hw_ver = ( ((v>>24) & 255), ((v>>16) & 255), ((v>>8) & 255) )
        return "%d.%d.%d" % hw_ver

    @property
    def model_id(self):
        return hex(self._model)

    @property
    def software_version(self):
        return "%d.%d.%d.%d" % self.__software_version

    @property
    def processing_steps(self):
        return self._processing_steps
    @processing_steps.setter
    def processing_steps(self, value):
        _logger.debug("Set processing steps")
        if value != int(self._processing_steps):
            self._processing_steps = value & self._available_processing_steps
            self.__write_integer(_CommandCodes.SetProcessingSteps, int(self._processing_steps))

    # --- Exposure

    @property
    def exposure_time(self):
        return self.__exposure_time
    @exposure_time.setter
    def exposure_time(self, value):
        _logger.info("New exposure time will be " + str(value))
        if self.status != SpectrometerStatus.Idle:
            raise IOError("Spectrometer is not idle")
        if value > self._max_exposure_time:
            _logger.error("New exposure time = " + str(value) + "us > maximum exposure time = " + str(self._max_exposure_time) + "us")
            raise ValueError("New exposure time is too large")
        if value < self._min_exposure_time:
            _logger.error("New exposure time = " + str(value) + "us < minimum exposure time = " + str(self._min_exposure_time) + "us")
            raise ValueError("New exposure time is too small")
        self.__exposure_time = value
        self.__write_integer(_CommandCodes.SetExposureTime, int(self.__exposure_time * 1e6)) # also cancels exposure
        # onExposureTimeChanged()

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
        self.__write_integer(_CommandCodes.SetAveraging, value)
        #onaveragingchanged()

    def start_exposure(self, num_exposures=None):
        """Starts the exposure.
        num_exposures is the number of exposures
        OR -1 for continuous exposure (keeping only the most recent spectrum)
        OR -2 for continuous exposure (keeping all spectra, unless FIFO buffer overflows).
        This method also discards any spectrum that is available but hasn't been read out yet."""
        if num_exposures is None:
            self.__write_integer(_CommandCodes.StartExposure, 1)
        else:
            _logger.info("Start exposure of n=" + str(num_exposures) + " spectras. (Negative value means continuous exposure.)")
            self.__write_integer(_CommandCodes.StartExposure, num_exposures)

    def cancel_exposure(self):
        if not self._isopen:
            self.__write_command(_CommandCodes.CancelExposure)

    @property
    def available_spectra(self):
        return self.__read_integer(_CommandCodes.GetStatus) >> 8

    @property
    def status(self):
        if self._isopen:
            return (self.__read_integer(_CommandCodes.GetStatus) & 255)
        else:
            return SpectrometerStatus.Closed

    def get_spectrum_data(self):
        _logger.debug("Get spectrum data...")
        bytedata = self.__read_data(_CommandCodes.Get32BitSpectrum)
        data = calibratedspectrometer.SpectrumData()
        spec_buffer = [0.0] * self._pixel_count
        data.Spectrum = spec_buffer
        p = 48 # header size - return code = 52-4
        if struct.unpack("<H", bytedata[20:22])[0] != self._pixel_count:
            raise ValueError("Pixel number from device does not match.")
            _logger.error("We got " + str(struct.unpack("<H", bytedata[20:22])[0]) + "pixels but device has " + str(self._pixel_count) + " pixels")
        pixel_format = struct.unpack("<B", bytedata[22:23])[0]
        _logger.debug("pixelFormat = " + str(pixel_format))
        if pixel_format > 3:
            raise ValueError("PixelFormat " + str(pixel_format) + " not supported.")

        if pixel_format == 0: #32BitFloatMetaDataUncompressed
            for i in range(self._pixel_count):
                spec_buffer[i] = (struct.unpack("<f", bytedata[p:p+4])[0])
                p += 4
        else:
            raise NotImplementedError("PixelFormat " + str(pixel_format) + "not supported.")

        # SPECTRUM_HEADER:
        #  0 int ExposureTime   # in us
        #  4 int Avaraging
        #  8 uint Timestamp     # in 1 ms units, start of exposure
        # 12 float LoadLevel
        # 16 float Temperature  # in degree Celcius
        # 20 uint16_t PixelCount
        # 22 uint16_t PixelFormat       # see below
        # 24 uint16_t ProcessingSteps   # applied processing steps
        # 26 uint16_t IntensityUnit
        # 28 int Spectrum Dropped   # not implemented
        # 32 float SaturationValue
        # 36 float OffsetAvg
        # 40 float DarkAvg
        # 44 float ReadoutNoise

        # Fill header of SpectrumData
        # ToDo: Get time stamp, convert from SysTick to datetime
        data.TimeStamp = datetime.datetime.now()
        data.ExposureTime = round(struct.unpack("<I", bytedata[0:4])[0] * 1e-6, 6)
        data.Averaging = struct.unpack("<I", bytedata[4:8])[0]
        data.LoadLevel = struct.unpack("<f", bytedata[12:16])[0]
        data.Temperature = struct.unpack("<f", bytedata[16:20])[0]
        data.AppliedProcessingSteps = struct.unpack("<H", bytedata[24:26])[0]
        data.IntensityUnit = struct.unpack("<H", bytedata[26:28])[0]
        data.SaturationValue = struct.unpack("<f", bytedata[32:36])[0]
        data.OffsetAvg = struct.unpack("<f", bytedata[36:40])[0]
        data.DarkAvg = struct.unpack("<f", bytedata[40:44])[0]
        data.ReadoutNoise = struct.unpack("<f", bytedata[44:48])[0]
        # ToDo: corrections necessary when processed on-board?
        self._load_level = data.LoadLevel
        self._time_stamp = data.TimeStamp

        _logger.debug("LoadLevel = " + str(self._load_level))
        _logger.debug("TimeStamp = " + str(self._time_stamp))

        return data

    # --- Temperature control

    @property
    def can_read_temperature(self):
        return True

    @property
    def temperature(self):
        t = self.__read_float(_CommandCodes.GetTemperature)
        if ( t < -30.0 or t > 80.0 ):
            _logger.debug("Device temperature: " + str(t) + " degC does not seem to be correct.")
        return t

    # --- Load and save calibration

    def get_wavelengths(self): # this overrides method in spectrometer.py
        _logger.info("Get wavelengths (in nm)...")
        if self.__software_version < (2, 1, 0, 1): # FixMe: also check for new calibration data format version?
            # Do not raise exception. Try to use parent method that might return default values.
            _logger.warning("Attempt to use parent method to get wavelengths. This might return default values. Please update your device's firmware to support this feature.")
            return super(Qseries, self).get_wavelengths() # Python 2 equivalent new-style classes (for compability reasons)
        else:
            bytedata = self.__read_data(_CommandCodes.GetWavelengths)
            # Unpacking with correct pixel count ensured by reading bulk data
            unpack_format = "<" + str(self._pixel_count) + "f"
            _logger.debug("Unpacking wavelengths with format \"" + unpack_format + "\"")
            lambda_nm = struct.unpack(unpack_format, bytedata)
            return lambda_nm

    @property
    def wavelength_coefficients(self): # This overrides method in spectrometer.py
        _logger.info("Get wavelength coefficients...")
        if self.__software_version < (2, 1, 4, 0):
            raise ValueError("Not supported by firmware version < 2.1.4")
        else:
            bytedata = self.__read_data(_Cmd.BulkData | _Cmd.Get | 0x05)
            self._wavelength_coefficients = struct.unpack("<4f", bytedata) # length is always four floats
            return self._wavelength_coefficients

    @property
    def nonlinearity_coefficients(self):
        """Gets the nonlinearity coefficients."""
        _logger.info("Get nonlinearity coefficients...")
        if self.__software_version < (2, 1, 4, 0): # return default values
            raise ValueError("Not supported by firmware version < 2.1.4")
        else:
            bytedata = self.__read_data(_Cmd.BulkData | _Cmd.Get | 0x06)

            numCoeff = struct.unpack_from("<I", bytedata)[0]
            _logger.debug("numCoeff = " + str(numCoeff))
            unpack_format = "<" + str(numCoeff) + "f"
            _logger.debug("Unpacking nonlinearity coefficients with format \"" + unpack_format + "\"")
            self._nonlinearity_coefficients = struct.unpack_from(unpack_format, bytedata[4:])
            return self._nonlinearity_coefficients  # ToDo: later return list of lists with numCoeff?


    def load_user_calibration(self):
        self.__load_calibration(True, True, True, True, 0)

    def restore_factory_calibration(self, restore_wavelengths, restore_nonlinearity, restore_dark_spectra, restore_spectral_sensitivity):
        raise NotImplementedError("Method not implemented yet.")

    def __load_calibration(self, wavelengths, nonlinearity, dark, sensitivity, start_page):
        if self.status != SpectrometerStatus.Idle:
            raise IOError("Spectrometer is not idle")
        _logger.info("Loading calibration data...")

        bufferpos = self.__CALIBR_PAGE_SIZE # start with reading first page
        bytedata = None
        page = 0
        length = 1
        bytebuffer = None

        i = 0
        while i < length:
            if bufferpos == self.__CALIBR_PAGE_SIZE:
                bytebuffer = self.__read_data(_CommandCodes.GetCalibrationData, start_page + page)
                _logger.debug(str(len(bytebuffer)) + " bytes for page " + str(start_page + page))
                if page == 0: # if first page: read length and create bytedata array
                    length, = struct.unpack("<I", bytebuffer[0:4])
                    _logger.debug("calibration data length: " + str(length) + " bytes")
                    if (length < 128 or length > self.__CALIBR_PAGE_SIZE * self.__calibr_num_pages):
                        _logger.info("No Calibration data available")
                        break # if no calibration data available: exit reading, bytedata remains None
                    bytedata = [0] * length
                    bufferpos = 4 # continue after first integer (which is length of calibration data)
                else:
                    bufferpos = 0

                page += 1

            bytedata[i] = bytebuffer[bufferpos]
            bufferpos += 1
            i += 1

        self._deserialize_calibration_data(bytedata, wavelengths, nonlinearity, dark, sensitivity)
        _logger.info("Loading calibration data... Done")

    @property
    def __factory_calibration_available(self):
        bytebuffer = self.__read_data(_CommandCodes.GetCalibrationData, self.__calibr_num_pages)
        length, = struct.unpack("<I", bytebuffer[0:4]) # first integer is length
        return (length >= 128 and length <= self.__CALIBR_PAGE_SIZE * self.__calibr_num_pages)

    def save_user_calibration(self):
        raise NotImplementedError("Method not implemented yet.")

    def save_factory_calibration(self):
        raise NotImplementedError("Method not implemented yet.")

    def save_calibration(self, start_page):
        raise NotImplementedError("Method not implemented yet.")

    def load_user_data(self):
        if self.__userdata_num_pages == 0:
            raise ValueError("Device cannot store user data.")
        raise NotImplementedError("Method not implemented yet.")
        return self.__read_data(_CommandCodes.GetUserData, 0)

    def save_user_data(self, data):
        raise NotImplementedError("Method not implemented yet.")

    # --- Digital I/O Ports

    def __initialize_port(self):
        self._num_io_pins = 4
        pincfg = self.__read_integer(_CommandCodes.GetPortConfig) # Byte x = Pin x
        for i in range(self._num_io_pins):
            self.__pin_config[i] = pincfg & 0xFF
            pincfg = pincfg // 256

        triggercfg = self.__read_integer(_CommandCodes.GetTriggerConfiguration)
        # Byte 0 = SpectrometerTriggerOptions
        # Byte 1 = RisingEdge
        # Byte 2 = Trigger pin
        self.__trigger_pin = (triggercfg >> 16) & 0xFF
        self._external_trigger_rising_edge = ( ((triggercfg >> 8) & 0xFF) != 0 )
        self.__trigger_option = triggercfg & 0xFF
        self.__use_trigger = self.__read_integer(_CommandCodes.GetTriggerEnabled) != 0
        self._trigger_option_available[1] = True
        # ToDo: implement low-jitter-mode and set _trigger_option_available[2] = True

    def set_io_pin_configuration(self, pin, config):
        if ( pin < 0 or pin > self._num_io_pins - 1 ):
            raise ValueError("Port number out of range.")
        self.__pin_config[pin] = config

        pincfg = int(self.__pin_config[0])
        + 256 * int(self.__pin_config[1])
        + 256 * 256 * int(self.__pin_config[2])
        + 256 * 256 * 256 * int(self.__pin_config[3])
        self.__write_integer(_CommandCodes.SetPortConfig, pincfg)
        _logger.debug("Port config is now: " + str(hex(self.__read_integer(_CommandCodes.GetPortConfig))))
        return True
    def get_io_pin_configuration(self, pin):
        if (pin < 0 or pin > self._num_io_pins - 1):
            raise ValueError("Pin number out of range.")
        return self.__pin_config[pin]

    @property
    def io_pins(self):
        return self.__read_integer(_CommandCodes.ReadPort) # Bit x = Pin x

    @property
    def external_trigger_source(self):
        return self.__trigger_pin
    @external_trigger_source.setter
    def external_trigger_source(self, value):
        if (value < 0 or value > self._num_io_pins -1):
            raise ValueError("Pin number out of range")
        self.__trigger_pin = value
        self.__send_trigger_mode()

    @property
    def trigger_option(self):
        return self.__trigger_option
    @trigger_option.setter
    def trigger_option(self, value):
        self.__trigger_option = value
        self.__send_trigger_mode()

    @property
    def external_trigger_rising_edge(self):
        return self._external_trigger_rising_edge
    @external_trigger_rising_edge.setter
    def external_trigger_rising_edge(self, value):
        self._external_trigger_rising_edge = value
        self.__send_trigger_mode()

    def __send_trigger_mode(self):
        # Byte 0 = SpectrometerTriggerOptions, Byte 1 = RisingEdge, Byte 2 = Trigger pin
        triggercfg = int(self.__trigger_option) + (1 if self._external_trigger_rising_edge else 0) * 256 + self.__trigger_pin * 65536
        self.__write_integer(_CommandCodes.SetTriggerConfiguration, triggercfg)

    @property
    def use_external_trigger(self):
        return self.__use_trigger
    @use_external_trigger.setter
    def use_external_trigger(self, value):
        if value == self.__use_trigger:
            return
        self.__write_integer(_CommandCodes.SetTriggerEnabled, (1 if value else 0))
        self.__use_trigger = value

    @property
    def canuse_external_trigger(self):
        return True

    # --- Device Access

    __max_rx_data_length = 16384

    ## Interface Layer

    def __send_receive_data(self, tx_data, length):
        """In principal this method is an abstraction of the 'transport layer'.
        tx_data and rx_data_buffer can be the same."""
        if not self._isopen:
            raise IOError("Device connection is closed.")
        # ToDo: later support also other serial interfaces
        self.usbWrite(tx_data)
        rx_data_buffer = self.usbRead(self.__max_rx_data_length)

        return rx_data_buffer

    ## Message Layer (currently only binary communication)

    def __write_command(self, command):
        _logger.debug("Write Command 0x%04x", command)
        data_buffer = struct.pack("<I", command)
        rx_buffer = self.__send_receive_data(data_buffer, 4)
        if len(rx_buffer) != 4:
            raise IOError(str(len(rx_buffer)) + " instead of 4 bytes received.")
        if (rx_buffer[0] != 0):
            raise ValueError("Error code " + str(rx_buffer[0]) + " received from device.")

    def __write_integers(self, command, value1, value2):
        data_buffer = struct.pack("<Iii", command, value1, value2) # values are signed integers
        data_buffer = self.__send_receive_data(data_buffer, 12)
        if len(data_buffer) != 4:
            raise IOError(str(len(data_buffer)) + " instead of 4 bytes received.")
        if (data_buffer[0] != 0):
            raise ValueError("Error code " + str(data_buffer[0]) + " received from device.")

    def __write_integer(self, command, value):
        _logger.debug("Write %d with Command 0x%04x", value, command)
        data_buffer = struct.pack("<Ii", command, value) # value is signed integer
        data_buffer = self.__send_receive_data(data_buffer, 8)
        if len(data_buffer) != 4:
            raise IOError(str(len(data_buffer)) + " instead of 4 bytes received.")
        if data_buffer[0] != 0:
            raise ValueError("Error code " + str(data_buffer[0]) + " received from device.")

    def __read_integer(self, command):
        data_buffer = struct.pack("<I", command)
        data_buffer = self.__send_receive_data(data_buffer, 4)
        if len(data_buffer) < 4:
            raise IOError("Only " + str(len(data_buffer)) + " bytes received.")
        if data_buffer[0] != 0:
            raise ValueError("Error code " + str(data_buffer[0]) + " received from device.")

        value = struct.unpack("<i", data_buffer[4:]) # return signed integer
        _logger.debug("%d received from Command 0x%04x", value[0], command)
        return value[0]

    def __read_float(self, command):
        data_buffer = struct.pack("<I", command)
        data_buffer = self.__send_receive_data(data_buffer, 4)
        if len(data_buffer) < 4:
            raise IOError("Only " + str(len(data_buffer)) + " bytes received.")
        if data_buffer[0] != 0:
            raise ValueError("Error code " + data_buffer[0] + " received from device.")
        value = struct.unpack("<f", data_buffer[4:])
        _logger.debug("%f received from Command 0x%04x", value[0], command)
        return value[0]

    def __read_string(self, command):

        data_buffer = struct.pack("<I", command)
        data_buffer = self.__send_receive_data(data_buffer, 4)
        if len(data_buffer) < 4:
            raise IOError("Only n bytes received.")
        if data_buffer[0] != 0:
            raise ValueError("Error code " + data_buffer[0] + " received from device.")
        if data_buffer[len(data_buffer) -1] == 0:
            data_buffer = data_buffer # ToDo: strip null-terminator
        return "" # FixMe: return substring from parsed characters

    def __write_data(self, command, parameter, data):
        raise NotImplementedError("Method not yet implemented")

    def __read_data(self, command, parameter=None):
        if parameter is None:
            data_buffer = struct.pack("<I", command)
            data_buffer = self.__send_receive_data(data_buffer, 4)
            _logger.debug("%d bytes received from Command 0x%04x", len(data_buffer), command)
        else:
            data_buffer = struct.pack("<Ii", command, parameter) # parameter is signed integer
            data_buffer = self.__send_receive_data(data_buffer, 8)
            _logger.debug("%d bytes received from Command 0x%04x with Parameter %d", len(data_buffer), command, parameter)

        if len(data_buffer) < 4:
            raise IOError("Only " + str(len(data_buffer)) + " bytes received.")
        if data_buffer[0] != 0:
            raise ValueError("Error code " + str(data_buffer[0]) + " received from device.")
        data = data_buffer[4: len(data_buffer)] # FixMe: return dataBuffer directly without copying
        return data



# -- CommunicationInterface
#class CommunicationInterface(object):

    # Class attributes
    __timeout = 1000
    __readPipe = 0x81   # IN endpoint
    __writePipe = 0x01  # OUT endpoint

    def usbOpen(self):

        try:
            self.dev = self.__usbdev
        except:
            raise IOError("Error while searching for device.")

        if self.dev is None:
            sys.exit("Could not find device")

        if self.dev.is_kernel_driver_active(0):
            try:
                self.dev.detach_kernel_driver(0)
                _logger.info("Kernel driver detached")
            except usb.core.USBError as e:
                sys.exit("Could not detach kernel driver: Can not use device ")
        else:
            _logger.info("No kernel driver attached")

        # Do not set the configuration if our desired configuration is already active.
        # This prevents a unintentional lightweight device reset.
        cfg = self.dev.get_active_configuration()
        cfg_desired = 1 # Qseries configuration is '1'

        if cfg is None:
            _logger.info("Device not configured. Set device configuration to %d." % cfg_desired)
            self.dev.set_configuration(cfg_desired)
        elif cfg.bConfigurationValue != cfg_desired:
            _logger.info("Device not configured with desired configuration. Set device configuration to %d." % cfg_desired)
            self.dev.set_configuration(cfg_desired)
        else:
            _logger.info("Device already configured with desired configuration. Device configuration is %d." %cfg.bConfigurationValue)
            pass

        usb.util.claim_interface(self.dev, 0) # FixMe: [Errno 16] Resource busy also indicates another instance is already running

    def usbClose(self):
        #if isOpen # of usb interface
        usb.util.dispose_resources(self.dev)
        self.dev = None
        self.__usbdev = None
        #isOpen = False # of usb interface
        pass

    def usbRead(self, size): # size_or_buffer
        if not self._isopen:
            raise IOError("USB connection is closed.")
        rxBuf = self.dev.read(self.__readPipe, self.__max_rx_data_length, self.__timeout)

        return rxBuf

    # ToDo: how to handle zero length packet (host-to-device transfer when transfer is multiple of endpoints wMaxPacketSize)
    def usbWrite(self, txBuf):
        if not self._isopen:
            raise IOError("USB connection is closed.")
        bytesWritten = self.dev.write(self.__writePipe, txBuf, self.__timeout)
        if bytesWritten != len(txBuf):
            _logger.error("Device write failed") # (" + bytesWritten + " of " + len(txBuf) + " bytes sent, " + errormessage ")")
            raise IOError("Device write failed")

        # ToDo: Try once more


## New Command Codes
_Cmd = enum(
        # MsgType
        Command = 0x0000, # if applicable with separate Parameter
        Parameter = 0x1000,
        DeviceProperty = 0x2000,
        MeasuredValue = 0x3000,
        BulkData = 0x4000, # if applicable with Index as separate Parameter

        # MsgKind
        Get = 0x000,
        Set = 0x100,
        Min = 0x200,
        Max = 0x300,
        Def = 0x400,
        Type = 0x800,
        Name = 0x900,
        Unit = 0xA00,
        Length = 0xF00,

        # Commands
        # ...for all devices
        Initialize = 0x00, # start communication; wake up device
        Bye = 0x01, # stop communication; send device to sleep
        SystemReset = 0x02, # perform reset
        ParameterReset = 0x03,
        # ...for spectrometer
        StartExposure = 0x04, # with parameter number of exposures
        CancelExposure = 0x05,

        # MeasuredValues
        Status = 0x00,
        SensorTemperature = 0x01,
        IOPortStatus = 0x02,
        SysTick = 0x03,
        RemainingExposures = 0x04,
        BufferCount = 0x05,

        # BulkData
        Spectrum = 0x00, # read only
        Wavelengths = 0x01, # read only
        CalibrationData = 0x02, # read/write, with index
        UserData = 0x03, # read/write, with index
        AuxInterface = 0x04, # read/write, and parameters

        # DeviceProperties
        DeviceID = 0x00,
        SerialNo = 0x01,
        Manufacturer = 0x02,
        ModelName = 0x03,
        HardwareVersion = 0x04,
        SoftwareVersion = 0x05,
        SpectrumMaxValue = 0x06,
        PixelCount = 0x07,
        DataCount = 0x08,
        FirstOffsetPixel = 0x09,
        NumOffsetPixels = 0x0A,
        FirstDarkPixel = 0x0B,
        NumDarkPixels = 0x0C,
        FirstRealPixel = 0x0D,
        PixelsPerBinExponent = 0x0E,
        MirrorSpectrum = 0x0F,
        SensorType = 0x10,
        OpticalConfiguration = 0x11,
        BadPixels0 = 0x16,
        BadPixels1 = 0x17,
        BadPixels2 = 0x18,
        BadPixels3 = 0x19,
        CalibrDataNumPages = 0x1A,
        UserDataNumPages = 0x1B,
        ReadoutNoise = 0x1C,

        # Parameters
        ExposureTime = 0x00,
        Averaging = 0x01,
        ProcessingSteps = 0x02,
        IOConfiguration = 0x03,
        TriggerConfiguration = 0x04,
        TriggerDelay = 0x05,
        ExternalTriggerEnable = 0x06,
    )

_CommandCodes = enum( # private! ToDo: use namedtuples?
        GetDeviceID = _Cmd.DeviceProperty | _Cmd.Get | _Cmd.DeviceID,
        GetSerialNo = _Cmd.DeviceProperty | _Cmd.Get | _Cmd.SerialNo,
        GetManufacturer = _Cmd.DeviceProperty | _Cmd.Get | _Cmd.Manufacturer,
        GetModelname = _Cmd.DeviceProperty | _Cmd.Get | _Cmd.ModelName,
        GetExposureTime = _Cmd.Parameter | _Cmd.Get | _Cmd.ExposureTime,
        SetExposureTime = _Cmd.Parameter | _Cmd.Set | _Cmd.ExposureTime,
        GetMinExposureTime = _Cmd.Parameter | _Cmd.Min | _Cmd.ExposureTime,
        GetMaxExposureTime = _Cmd.Parameter | _Cmd.Max | _Cmd.ExposureTime,
        GetDataCount = _Cmd.DeviceProperty | _Cmd.Get | _Cmd.DataCount,
        GetMaxDataValue = _Cmd.DeviceProperty | _Cmd.Get | _Cmd.SpectrumMaxValue,
        GetRawData = 0x3B,
        Reset = _Cmd.Command | _Cmd.Initialize,
        Bye = _Cmd.Command | _Cmd.Bye,
        SystemReset = _Cmd.Command | _Cmd.SystemReset,
        StartExposure = _Cmd.Command | _Cmd.StartExposure,
        CancelExposure = _Cmd.Command | _Cmd.CancelExposure,
        GetStatus = _Cmd.MeasuredValue | _Cmd.Status,
        Get32BitSpectrum = _Cmd.BulkData | _Cmd.Status,
        GetTemperature = _Cmd.MeasuredValue | _Cmd.Get | _Cmd.SensorTemperature,

        GetProcessingSteps = _Cmd.Parameter | _Cmd.Get | _Cmd.ProcessingSteps,
        SetProcessingSteps = _Cmd.Parameter | _Cmd.Set | _Cmd.ProcessingSteps,
        GetMaxProcessingSteps = _Cmd.Parameter | _Cmd.Max | _Cmd.ProcessingSteps,
        GetDefaultProcessingSteps = _Cmd.Parameter | _Cmd.Def | _Cmd.ProcessingSteps,
        GetHardwareVersion = _Cmd.DeviceProperty | _Cmd.Get | _Cmd.HardwareVersion,
        GetSoftwareVersion = _Cmd.DeviceProperty | _Cmd.Get | _Cmd.SoftwareVersion,

        GetPixelCount = _Cmd.DeviceProperty | _Cmd.Get | _Cmd.PixelCount,
        GetFirstPixel = _Cmd.DeviceProperty | _Cmd.Get | _Cmd.FirstOffsetPixel,
        GetFirstOffsetPixel = _Cmd.DeviceProperty | _Cmd.Get | _Cmd.FirstOffsetPixel,
        GetNumOffsetPixels = _Cmd.DeviceProperty | _Cmd.Get | _Cmd.NumOffsetPixels,
        GetFirstDarkPixel = _Cmd.DeviceProperty | _Cmd.Get | _Cmd.FirstDarkPixel,
        GetNumDarkPixels = _Cmd.DeviceProperty | _Cmd.Get | _Cmd.NumDarkPixels,
        GetMirrorSpectrum = _Cmd.DeviceProperty | _Cmd.Get | _Cmd.MirrorSpectrum,
        GetSensorType = _Cmd.DeviceProperty | _Cmd.Get | _Cmd.SensorType,
        GetAveraging = _Cmd.Parameter | _Cmd.Get | _Cmd.Averaging,
        SetAveraging = _Cmd.Parameter | _Cmd.Set | _Cmd.Averaging,
        GetMaxAveraging = _Cmd.Parameter | _Cmd.Max | _Cmd.Averaging,

        SetPortConfig = _Cmd.Parameter | _Cmd.Set | _Cmd.IOConfiguration,
        GetPortConfig = _Cmd.Parameter | _Cmd.Get | _Cmd.IOConfiguration,
        ReadPort = _Cmd.MeasuredValue | _Cmd.Get | _Cmd.IOPortStatus,
        SetTriggerConfiguration = _Cmd.Parameter | _Cmd.Set | _Cmd.TriggerConfiguration,
        GetTriggerConfiguration = _Cmd.Parameter | _Cmd.Get | _Cmd.TriggerConfiguration,
        SetTriggerEnabled = _Cmd.Parameter | _Cmd.Set | _Cmd.ExternalTriggerEnable,
        GetTriggerEnabled = _Cmd.Parameter | _Cmd.Get | _Cmd.ExternalTriggerEnable,
        GetNumPorts = 0x49,

        GetWavelengths = _Cmd.BulkData | _Cmd.Get | _Cmd.Wavelengths,
        GetCalibrationData = _Cmd.BulkData | _Cmd.Get | _Cmd.CalibrationData,
        GetUserData = _Cmd.BulkData | _Cmd.Get | _Cmd.UserData,
        SetUserData = _Cmd.BulkData | _Cmd.Set | _Cmd.UserData,
        GetCalibrationDataNumPages = _Cmd.DeviceProperty | _Cmd.Get | _Cmd.CalibrDataNumPages,
        GetUserDataNumPages = _Cmd.DeviceProperty | _Cmd.Get | _Cmd.UserDataNumPages,

        GetSysTick = _Cmd.MeasuredValue | _Cmd.Get | _Cmd.SysTick,
        ParameterReset = _Cmd.Command | _Cmd.ParameterReset,
        )

_RetCodes = enum( # private (do not export)
        OK = 0,
        UnknownCommandCode = 1,
        InvalidParameter = 2, # e.g. out of range
        MissingParameter = 3,
        InvalidOperation = 4, # user-fixable (e.g. current state of the device does not allow this operation)
        NotSupported = 5,     # This device does not support this operation.
        PasscodeInvalid = 6,  # A passcode required for certain operations was wrong or not given.
        CommunicationError = 7, # protocol violations or time-outs
        InternalError = 8,    # non user-fixable -> call support
        UnknownBootloaderCommandCode = 9, # If the bootloader is active: This is used instead of 1 (UnknownCommandCode) to give the user a hint that the bootloader is active.
        )

# ToDo: map return codes to strings
_str_retcodes_map = {
            _RetCodes.OK:'Success (no error)',
            _RetCodes.UnknownCommandCode:'Unknown command code',
            _RetCodes.InvalidParameter:'Invalid parameter',
            }


