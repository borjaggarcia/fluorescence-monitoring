#!/usr/bin/env python


from __future__ import print_function

__author__ = "RGB Photonics GmbH"

"""
Simple Spectrometer

A demo program to take a spectrum with a new Qseries spectrometer
demonstrating how to use the rgbdriverkit in order to use the
spectrometer from your own software.

Note: For plotting/saving the graph matplotlib is required.

Copyright 2017 RGB Photonics GmbH
written by Martin Hofmann
Version 0.1.1 - April 28, 2017
"""

import sys
import time

# Uncomment the following when not using setup.py or pip to install the packages
# Replace the path to the appropriate package location
#sys.path.append('/path/to/pyrgbdriverkit/')
#sys.path.append('/path/to/pyusb/')

import rgbdriverkit as rgbdriverkit
from rgbdriverkit.qseriesdriver import Qseries
from rgbdriverkit.spectrometer import SpectrometerStatus
from rgbdriverkit.calibratedspectrometer import SpectrumData
from rgbdriverkit.calibratedspectrometer import SpectrometerProcessing


def main(save_plot=None):

    print("Demo program started.")
    print("rgbdriverkit version: " + rgbdriverkit.__version__)

    dev = Qseries.search_devices()
    if (dev != None):
        print("Device found.")
    else:
        sys.exit("No device found.")

    q = Qseries(dev[0]) # Create instance of first spectrometer found

    # Print device properties
    print("Model name:", q.model_name)
    print("Manufacturer:", q.manufacturer)
    print("Serial Number:", q.serial_number)

    q.open() # Open device connection

    print("Software version: " + q.software_version)
    print("Hardware version: " + q.hardware_version)

    nm = q.get_wavelengths()

    # Set exposure time and start exposure
    q.exposure_time = 0.1 # in seconds
    print("Starting exposure with t=" + str(q.exposure_time) + "s")
    q.processing_steps = SpectrometerProcessing.AdjustOffset # only adjust offset
    q.start_exposure(1)
    print("Waiting for spectrum...")
    while not q.available_spectra:
        time.sleep(0.1)

    print("Spectrum available")
    spec = q.get_spectrum_data() # Get spectrum with meta data

    print("TimeStamp:", spec.TimeStamp)
    print("LoadLevel: %.2f" % spec.LoadLevel)
    print("ExposureTime: " + str(spec.ExposureTime) + "s")

    q.close() # Close device connection

    if save_plot is True:
        print("Plot spectrum and save figure to file 'spectrum.png'.")
        import matplotlib.pyplot as plt
        # Create plot
        plt.clf()
        plt.plot(nm, spec.Spectrum)
        plt.grid(True)
        plt.title('Spectrum')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('ADCvalues')
        plt.draw()
        plt.show()
        # save figure to file
        plt.savefig("spectrum.png")

    print("Done")

if __name__ == "__main__":
    main(save_plot=False)
