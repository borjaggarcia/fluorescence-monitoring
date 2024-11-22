# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------------------
# Title: Fluorescence Monitoring Script for Pioreactor
# Description:
# This script provides a comprehensive solution for monitoring fluorescence within a Pioreactor system.
# It controls LED intensities, manages periodic spectrometer measurements, and interfaces with an Arduino
# for LED manipulation.
#
# Key Features:
# - **LED Control**: Sets LED intensities based on user-defined values and can control LEDs via an Arduino.
# - **Spectrometer Measurements**: Performs two consecutive spectrometer measurements with different
#   integration times (0.1s and 5s) for each excitation LED.
# - **Data Processing**: Estimates peak intensity at 5s based on 0.1s measurement and normalizes fluorescence peaks.
# - **Data Management**: Saves spectra in separate folders based on integration time, logs intensity readings
#   at specific wavelengths, and persists data in an SQLite database for later retrieval and analysis.
# - **Safety & State Management**: Implements a locking mechanism for hardware access to avoid conflicts
#   and handles various job states to ensure stable operation and data accuracy.
#
# Plugin Metadata:
# - Plugin Name: Fluorescence Monitoring
# - Version: 3.0
# - Author: Borja Garcia Garcia
# -------------------------------------------------------------------------------------------------

# --- IMPORTS -------------------------------------------------------------------------------------
import time
import serial
import click
import threading
import sys
import csv
import os
import base64
import json
import sqlite3  # To save data to the SQLite database
from threading import Lock
import logging
import numpy as np  # Added import for numpy
from datetime import datetime
import pytz  # Added for timezone handling

from pioreactor.background_jobs.base import BackgroundJob
from pioreactor.actions.led_intensity import led_intensity
from pioreactor.whoami import get_unit_name, get_assigned_experiment_name

import rgbdriverkit
from rgbdriverkit.qseriesdriver import Qseries
from rgbdriverkit.calibratedspectrometer import SpectrumData, SpectrometerProcessing
# -------------------------------------------------------------------------------------------------

# --- PLUGIN INFO ---------------------------------------------------------------------------------
__plugin_summary__ = 'Monitors fluorescence by controlling LED intensities, performs periodic spectrometer measurements, and manages Arduino communication for LED control.'
__plugin_version__ = '3.0'
__plugin_name__ = "Fluorescence Monitoring"
__plugin_author__ = "Borja Garcia Garcia"
# -------------------------------------------------------------------------------------------------

__all__ = ["FluorescenceMonitoring", "start_fluorescence_monitoring"]

# Create a shared hardware lock to prevent concurrent access
hardware_lock = Lock()

# --- SPECTROMETER MEASUREMENT --------------------------------------------------------------------
class SpectrometerMeasurement:
    """
    Class to perform a single spectrometer measurement.
    """

    def __init__(self, unit, experiment, excitation_wavelength, exposure_time=5, pub_client=None, logger=None, estimated_peak_intensity=None):
        self.exposure_time = float(exposure_time)
        self.excitation_wavelength = str(excitation_wavelength)
        self.pub_client = pub_client
        self.q = None
        self.unit = unit
        self.experiment = experiment
        self.logger = logger or self.get_default_logger()
        self.estimated_peak_intensity = estimated_peak_intensity  # Added for normalization
        self.logger.info(f'Initializing spectrometer measurement with exposure time {self.exposure_time} seconds and excitation wavelength {self.excitation_wavelength} nm.')

    def get_default_logger(self):
        logger = logging.getLogger('SpectrometerMeasurement')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s %(levelname)s [%(name)s] %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def connect_to_spectrometer(self):
        dev = Qseries.search_devices()
        if not dev:
            self.logger.error('No spectrometer device found.')
            raise Exception('No spectrometer device found.')
        self.q = Qseries(dev[0])
        self.q.open()
        self.logger.info(f'Connected to spectrometer: {self.q.model_name} (Serial: {self.q.serial_number})')

    def run(self):
        """
        Run the spectrometer measurement and collect spectrum data.
        Returns the estimated peak intensity at 5s if exposure_time is 0.1s,
        or normalized intensities if exposure_time is 5s.
        """
        self.connect_to_spectrometer()
        try:
            nm = self.q.get_wavelengths()

            # Set exposure time and start exposure
            self.q.exposure_time = self.exposure_time
            self.logger.info(f'Starting exposure with t={self.exposure_time} seconds.')
            self.q.processing_steps = SpectrometerProcessing.AdjustOffset

            try:
                self.q.start_exposure(1)
            except Exception as e:
                self.logger.error(f'Failed to start exposure: {e}')
                return

            self.logger.info('Waiting for spectrum...')

            # Adding a timeout for the wait
            timeout = self.exposure_time + 10  # Timeout in seconds
            start_time = time.time()

            while not self.q.available_spectra:
                elapsed_time = time.time() - start_time
                self.logger.debug(f'No spectrum available yet, waiting... elapsed time: {elapsed_time}s')
                time.sleep(1)  # Increase sleep time for less CPU usage

                # Check for timeout
                if elapsed_time > timeout:
                    self.logger.error('Timeout while waiting for the spectrum. Exiting.')
                    return  # Exit the function if no spectrum is available after timeout

            self.logger.info('Spectrum available')

            try:
                spec = self.q.get_spectrum_data()
            except Exception as e:
                self.logger.error(f'Failed to get spectrum data: {e}')
                return

            # Convert 'nm' and 'spec.Spectrum' to numpy arrays
            nm_array = np.array(nm)
            spectrum_array = np.array(spec.Spectrum)

            # Calculate peak intensity
            peak_intensity = np.max(spectrum_array)
            self.logger.info(f'Peak intensity: {peak_intensity}')

            # If exposure_time is 0.1s, estimate peak intensity at 5s
            if self.exposure_time == 0.1:
                estimated_peak_intensity_at_5s = peak_intensity * (5 / 0.1)
                self.logger.info(f'Estimated peak intensity at 5s: {estimated_peak_intensity_at_5s}')
                # Save the spectrum
                self.save_spectrum_to_csv(nm, spec.Spectrum)
                # Return the estimated peak intensity at 5s
                return estimated_peak_intensity_at_5s
            elif self.exposure_time == 5:
                # Use the provided estimated peak intensity for normalization
                if self.estimated_peak_intensity is not None:
                    normalization_factor = self.estimated_peak_intensity
                    self.logger.info(f'Using estimated peak intensity for normalization: {normalization_factor}')
                else:
                    normalization_factor = 1  # Avoid division by zero
                    self.logger.warning('No estimated peak intensity provided for normalization.')

                # Calculate average intensity over 638-678 nm for intensity_658nm
                indices_658nm_range = np.where((nm_array >= 638) & (nm_array <= 678))[0]
                if indices_658nm_range.size > 0:
                    intensity_658nm = np.mean(spectrum_array[indices_658nm_range])
                else:
                    intensity_658nm = 0
                    self.logger.warning('No data points found in 638-678 nm range for intensity_658nm.')

                # Calculate average intensity over 680-720 nm for intensity_700nm
                indices_700nm_range = np.where((nm_array >= 680) & (nm_array <= 720))[0]
                if indices_700nm_range.size > 0:
                    intensity_700nm = np.mean(spectrum_array[indices_700nm_range])
                else:
                    intensity_700nm = 0
                    self.logger.warning('No data points found in 680-720 nm range for intensity_700nm.')

                self.logger.info(f'Average intensity for 658 nm (638-678 nm): {intensity_658nm}')
                self.logger.info(f'Average intensity for 700 nm (680-720 nm): {intensity_700nm}')

                # Normalize the intensities
                normalized_intensity_658nm = intensity_658nm / normalization_factor
                normalized_intensity_700nm = intensity_700nm / normalization_factor

                self.logger.info(f'Normalized intensity for 658 nm: {normalized_intensity_658nm}')
                self.logger.info(f'Normalized intensity for 700 nm: {normalized_intensity_700nm}')

                # Publish data
                self.publish_intensities(normalized_intensity_658nm, normalized_intensity_700nm)
                self.save_spectrum_to_csv(nm, spec.Spectrum)
                # Return normalized intensities
                return normalized_intensity_658nm, normalized_intensity_700nm
            else:
                self.logger.warning('Exposure time is not 0.1s or 5s, skipping processing.')

        except Exception as e:
            self.logger.error(f'Exception in spectrometer measurement: {e}')
            self.logger.exception(e)
        finally:
            if self.q:
                self.q.close()
                self.logger.info('Spectrometer connection closed.')

            # For exposure_time == 0.1s, return the estimated peak intensity at 5s
            if self.exposure_time == 0.1:
                return estimated_peak_intensity_at_5s

    def publish_intensities(self, intensity_658nm, intensity_700nm):
        """
        Publishes the normalized intensities at 658 nm and 700 nm via MQTT for real-time updates.
        """
        topic_658nm = f'pioreactor/spectrometer/normalized_intensity_658nm_{self.excitation_wavelength}nm'
        topic_700nm = f'pioreactor/spectrometer/normalized_intensity_700nm_{self.excitation_wavelength}nm'

        self.pub_client.publish(topic_658nm, str(intensity_658nm))
        self.logger.info(f'Published normalized intensity to MQTT topic: {topic_658nm}')

        self.pub_client.publish(topic_700nm, str(intensity_700nm))
        self.logger.info(f'Published normalized intensity to MQTT topic: {topic_700nm}')

    def save_spectrum_to_csv(self, wavelengths, intensities):
        """
        Saves the full spectrum (wavelengths and intensities) to a CSV file with excitation wavelength in filename.
        """
        base_output_dir = '/home/pioreactor/.pioreactor/storage/spectra_data'
        if not os.path.exists(base_output_dir):
            os.makedirs(base_output_dir)

        # Determine subfolder based on exposure_time
        if self.exposure_time == 0.1:
            subfolder = 't01'
        elif self.exposure_time == 5:
            subfolder = 't5'
        else:
            subfolder = 'other'  # In case other exposure times are used

        output_dir = os.path.join(base_output_dir, subfolder)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Set timezone
        tz = pytz.timezone('Europe/Madrid')
        # Get current UTC time and convert to local time zone
        utc_now = datetime.utcnow().replace(tzinfo=pytz.utc)
        local_time = utc_now.astimezone(tz)
        # Get timestamp in the desired format
        timestamp = local_time.strftime('%Y%m%d-%H%M%S')

        file_path = os.path.join(output_dir, f'spectrum_{self.excitation_wavelength}nm_{timestamp}.csv')

        try:
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Wavelength (nm)', 'Intensity'])  # Header
                for wavelength, intensity in zip(wavelengths, intensities):
                    writer.writerow([wavelength, intensity])

            self.logger.info(f'Spectrum saved to {file_path}')
        except Exception as e:
            self.logger.error(f'Failed to save spectrum to CSV: {e}')
# -------------------------------------------------------------------------------------------------

# --- FLUORESCENCE MONITORING  --------------------------------------------------------------------
class FluorescenceMonitoring(BackgroundJob):
    job_name = "fluorescence_monitoring"
    published_settings = {
        'interval': {'datatype': "integer", "unit": "seconds", "settable": True},
    }
    LED_channels = {"C": "C", "D": "D"}

    def __init__(self, unit, experiment, exposure_time='5', interval=900, arduino_port="/dev/arduino", **kwargs):
        super().__init__(unit=unit, experiment=experiment)
        self.exposure_time = float(exposure_time)
        self.interval = interval
        self.led_wavelengths = ["655", "620", "595", "527", "450", "405"]
        self.photoperiod_job_name = "led_automation"  # Name of the photoperiod automation job

        # Initialize hardware
        try:
            self.arduino = serial.Serial(arduino_port, 9600, timeout=1)
            self.logger.info(f"Connected to Arduino on port {arduino_port}")
        except serial.SerialException as e:
            self.logger.error(f"Failed to connect to Arduino on port {arduino_port}: {e}")
            raise

        # Connect to SQLite database
        self.connect_to_sql()

        # Start passive listeners
        self.start_passive_listeners()

        # Set initial state to READY
        self.set_state(self.READY)

    def connect_to_sql(self):
        """
        Connect to the SQLite database.
        """
        self.conn = sqlite3.connect('/home/pioreactor/.pioreactor/storage/pioreactor.sqlite')
        self.cursor = self.conn.cursor()
        self.logger.info('Connected to SQLite database.')

    def get_current_led_intensity_from_settings(self):
        """
        Retrieves the current LED intensity from the led_automation_settings table.
        """
        try:
            query = '''
                SELECT settings FROM led_automation_settings
                WHERE pioreactor_unit = ? AND experiment = ?
                ORDER BY started_at DESC LIMIT 1
            '''
            self.cursor.execute(query, (self.unit, self.experiment))
            result = self.cursor.fetchone()
            if result is not None:
                base64_encoded_settings = result[0]
                if base64_encoded_settings:
                    try:
                        # Decode the base64 string
                        decoded_bytes = base64.b64decode(base64_encoded_settings)
                        decoded_str = decoded_bytes.decode('utf-8')
                        # Parse the JSON
                        settings = json.loads(decoded_str)
                        self.logger.debug(f"Decoded settings: {settings}")
                        # Get the light intensity value
                        intensity = float(settings.get('light_intensity', 0.0))
                        self.logger.info(f"Retrieved light intensity: {intensity}%")
                        return intensity
                    except (base64.binascii.Error, json.JSONDecodeError) as decode_error:
                        self.logger.error(f"Error decoding settings: {decode_error}")
                        return 0.0
                else:
                    self.logger.warning("Settings field is empty, defaulting intensity to 0.")
                    return 0.0
            else:
                self.logger.warning("No settings found for led_automation, defaulting intensity to 0.")
                return 0.0
        except Exception as e:
            self.logger.error(f"Error querying settings for led_automation: {e}", exc_info=True)
            return 0.0

    def start_passive_listeners(self):
        # Subscribe to state changes
        self.subscribe_and_callback(
            self.on_state_change,
            f"pioreactor/{self.unit}/{self.experiment}/{self.job_name}/$state/set"
        )

    def on_state_change(self, client, userdata, message):
        new_state = message.payload.decode()
        if new_state == "sleeping":
            self.set_state(self.SLEEPING)
        elif new_state == "ready":
            self.set_state(self.READY)
        elif new_state == "disconnected":
            self.clean_up()

    def run(self):
        self.logger.info("Job starting the main cycle...")

        while self.state != self.DISCONNECTED:
            if self.state == self.SLEEPING:
                self.logger.info("Job is sleeping, waiting to resume")
                time.sleep(1)
                continue

            self.logger.info("Starting spectrometer measurement cycle")

            try:
                # Measure with spectrometer
                self.measure_with_spectrometer()
            except Exception as e:
                self.logger.error(f"Exception during spectrometer measurement: {e}")
                self.logger.exception(e)

            # Wait for the next cycle
            self.logger.info(f"Waiting {self.interval} seconds before the next cycle.")
            time.sleep(self.interval)

        self.logger.info("Job has been disconnected.")

    def measure_with_spectrometer(self):
        try:
            if self.state == self.DISCONNECTED:
                return

            with hardware_lock:
                self.logger.info("Acquired hardware lock for fluorescence measurement")

                # Pause ODReader job temporarily
                self.logger.info("Pausing ODReader job")
                self.publish(f"pioreactor/{self.unit}/{self.experiment}/od_reading/$state/set", "sleeping")
                # Wait until ODReader job is sleeping
                self.wait_for_job_state("od_reading", "sleeping", timeout=10)

                # Get current intensity from led_automation_settings
                intensity = self.get_current_led_intensity_from_settings()
                self.current_intensity_c = intensity
                self.current_intensity_d = intensity
                self.logger.info(f"Saved current intensities - C: {self.current_intensity_c}%, D: {self.current_intensity_d}%")

                # Turn off LEDs C and D for measurement
                self.logger.info("Turning off LEDs C and D for measurement")
                led_intensity({"C": 0, "D": 0})
                time.sleep(2)
                led_intensity({"C": 0, "D": 0})
                time.sleep(1)
                led_intensity({"C": 0, "D": 0})
                self.logger.info("Set LEDs C and D to 0% intensity for measurement.")
                time.sleep(1)  # Wait for LEDs to update

                # Pause photoperiod automation
                self.logger.info("Pausing photoperiod automation")
                self.publish(f"pioreactor/{self.unit}/{self.experiment}/{self.photoperiod_job_name}/$state/set", "sleeping")
                # Wait until photoperiod automation is sleeping
                self.wait_for_job_state(self.photoperiod_job_name, "sleeping", timeout=10)

                # Measurement cycle with spectrometer
                self.logger.info("Starting spectrometer measurement cycle")
                for led in self.led_wavelengths:
                    if self.state == self.SLEEPING or self.state == self.DISCONNECTED:
                        self.logger.info(f"Measurement paused or stopped, skipping LED {led} nm")
                        break

                    self.logger.info(f"Measuring with LED {led} nm")
                    self.activate_spectrometer(led)

                # Restore LEDs C and D to previous intensities
                led_intensity({"C": self.current_intensity_c, "D": self.current_intensity_d})
                self.logger.info(f"Restored LEDs C and D to previous intensities - C: {self.current_intensity_c}%, D: {self.current_intensity_d}%")

                # Resume ODReader job after measurement
                self.logger.info("Resuming ODReader job")
                self.publish(f"pioreactor/{self.unit}/{self.experiment}/od_reading/$state/set", "ready")
                # Wait until ODReader job is ready
                self.wait_for_job_state("od_reading", "ready", timeout=10)

                # Resume photoperiod automation
                self.logger.info("Resuming photoperiod automation")
                self.publish(f"pioreactor/{self.unit}/{self.experiment}/{self.photoperiod_job_name}/$state/set", "ready")
                # Wait until photoperiod automation is ready
                self.wait_for_job_state(self.photoperiod_job_name, "ready", timeout=10)

        except Exception as e:
            self.logger.error(f"Exception in measure_with_spectrometer: {e}")
            self.logger.exception(e)
        finally:
            pass  # Do not close the database connection here

    def wait_for_job_state(self, job_name, desired_state, timeout=10):
        """
        Waits until the specified job reaches the desired state.
        """
        start_time = time.time()
        current_state = None

        def on_state_message(client, userdata, message):
            nonlocal current_state
            current_state = message.payload.decode()

        topic = f"pioreactor/{self.unit}/{self.experiment}/{job_name}/$state"

        self.sub_client.subscribe(topic)
        self.sub_client.message_callback_add(topic, on_state_message)

        while current_state != desired_state:
            if time.time() - start_time > timeout:
                self.logger.warning(f"Timeout waiting for {job_name} to reach state '{desired_state}'")
                break
            time.sleep(0.1)

        self.sub_client.unsubscribe(topic)
        self.sub_client.message_callback_remove(topic)

    def activate_spectrometer(self, led_wavelength):
        try:
            # Send command to turn on the LED with the specific wavelength
            self.logger.info(f"Activating spectrometer with LED {led_wavelength} nm")
            self.send_arduino_command(f"leds_on [{led_wavelength}]")
            time.sleep(1)  # Wait for the LED to stabilize

            # Perform spectrometer measurements with different integration times
            self.logger.info(f"Starting spectrometer measurements with LED {led_wavelength} nm...")

            # First measurement at exposure time 0.1s
            spectrometer_measurement_01 = SpectrometerMeasurement(
                unit=self.unit,
                experiment=self.experiment,
                excitation_wavelength=led_wavelength,
                exposure_time=0.1,
                pub_client=self.pub_client,
                logger=self.logger
            )
            # Run spectrometer measurement and get estimated peak intensity at 5s
            estimated_peak_intensity_at_5s = spectrometer_measurement_01.run()

            if estimated_peak_intensity_at_5s is None:
                self.logger.warning("Estimated peak intensity at 5s is None, skipping normalization.")
                estimated_peak_intensity_at_5s = 1  # To avoid division by zero

            # Second measurement at exposure time 5s, passing the estimated peak intensity
            spectrometer_measurement_5 = SpectrometerMeasurement(
                unit=self.unit,
                experiment=self.experiment,
                excitation_wavelength=led_wavelength,
                exposure_time=5,
                pub_client=self.pub_client,
                logger=self.logger,
                estimated_peak_intensity=estimated_peak_intensity_at_5s  # Pass the estimated peak intensity
            )
            # Run spectrometer measurement and get normalized intensities
            result = spectrometer_measurement_5.run()
            if result is not None:
                normalized_intensity_658nm, normalized_intensity_700nm = result
                # Save to SQL using self.conn and self.cursor
                self.save_to_sql(normalized_intensity_658nm, normalized_intensity_700nm, led_wavelength)

            self.logger.info(f"Spectrometer measurements with LED {led_wavelength} nm completed successfully.")

        except Exception as e:
            self.logger.error(f"Error during spectrometer measurement: {e}")
            self.logger.exception(e)
        finally:
            # Turn off the LED after measurement
            self.deactivate_led(led_wavelength)
            time.sleep(2)

    def save_to_sql(self, intensity_658nm, intensity_700nm, excitation_wavelength):
        """
        Saves the normalized intensities at 658 nm and 700 nm to separate SQL tables for each excitation wavelength.
        """
        # Set timezone
        tz = pytz.timezone('Europe/Madrid')
        # Get current UTC time and convert to local time zone
        utc_now = datetime.utcnow().replace(tzinfo=pytz.utc)
        local_time = utc_now.astimezone(tz)
        # Get timestamp in the desired format
        timestamp = local_time.strftime('%Y-%m-%d %H:%M:%S')

        # Table names based on measurement and excitation wavelengths
        table_658nm = f'normalized_intensity_658nm_data_{excitation_wavelength}nm'
        table_700nm = f'normalized_intensity_700nm_data_{excitation_wavelength}nm'

        try:
            # Create table for normalized 658 nm intensity data if it doesn't exist
            self.cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {table_658nm} (
                    experiment TEXT NOT NULL,
                    pioreactor_unit TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    normalized_intensity REAL NOT NULL,
                    PRIMARY KEY (experiment, pioreactor_unit, timestamp)
                )
            ''')

            # Save normalized 658 nm intensity data
            self.cursor.execute(f'''
                INSERT INTO {table_658nm} (experiment, pioreactor_unit, timestamp, normalized_intensity)
                VALUES (?, ?, ?, ?)
            ''', (self.experiment, self.unit, timestamp, intensity_658nm))

            # Create table for normalized 700 nm intensity data if it doesn't exist
            self.cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {table_700nm} (
                    experiment TEXT NOT NULL,
                    pioreactor_unit TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    normalized_intensity REAL NOT NULL,
                    PRIMARY KEY (experiment, pioreactor_unit, timestamp)
                )
            ''')

            # Save normalized 700 nm intensity data
            self.cursor.execute(f'''
                INSERT INTO {table_700nm} (experiment, pioreactor_unit, timestamp, normalized_intensity)
                VALUES (?, ?, ?, ?)
            ''', (self.experiment, self.unit, timestamp, intensity_700nm))

            self.conn.commit()
            self.logger.info(f'Normalized data saved to SQL tables: {table_658nm} and {table_700nm}')
        except Exception as e:
            self.logger.error(f'Failed to save data to SQL: {e}')

    def deactivate_led(self, led_wavelength):
        self.logger.info(f"Turning off LED {led_wavelength} nm")
        try:
            self.send_arduino_command(f"leds_off [{led_wavelength}]")
        except Exception as e:
            self.logger.error(f"Failed to turn off LED {led_wavelength} nm: {e}")

    def send_arduino_command(self, command):
        """
        Sends a command to the Arduino and handles communication errors.
        """
        try:
            self.arduino.write(f"{command}\n".encode())
            # Optionally read response from Arduino to confirm receipt
            response = self.arduino.readline().decode().strip()
            self.logger.debug(f"Arduino response: {response}")
        except serial.SerialException as e:
            self.logger.error(f"Serial communication error: {e}")
            # Optionally implement reconnection logic here
            self.reconnect_arduino()

    def reconnect_arduino(self):
        """
        Attempts to reconnect to the Arduino in case of serial communication errors.
        """
        try:
            self.arduino.close()
            time.sleep(2)
            self.arduino.open()
            self.logger.info("Reconnected to Arduino.")
        except Exception as e:
            self.logger.error(f"Failed to reconnect to Arduino: {e}")

    def on_disconnected(self):
        # Clean up resources
        if self.arduino:
            try:
                self.arduino.close()
                self.logger.info("Arduino connection closed.")
            except Exception as e:
                self.logger.error(f"Error closing Arduino connection: {e}")
                self.logger.exception(e)
        # Close SQLite connection
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            self.logger.info('SQLite connection closed.')

        # Set LEDs C and D to 0% intensity on disconnect
        led_intensity({"C": 0, "D": 0})
        self.logger.info("Set LEDs C and D to 0% intensity on disconnect.")

    def set_interval(self, new_value):
        try:
            self.interval = int(new_value)
            self.logger.info(f"Interval updated to {self.interval} seconds.")
        except ValueError:
            self.logger.error(f"Invalid interval value: {new_value}")
# -------------------------------------------------------------------------------------------------

# --- CLICK ---------------------------------------------------------------------------------------
def start_fluorescence_monitoring(exposure_time='5', interval=900, arduino_port="/dev/arduino", unit=None, experiment=None):
    unit = unit or get_unit_name()
    experiment = experiment or get_assigned_experiment_name(unit)
    with FluorescenceMonitoring(unit, experiment, exposure_time, interval, arduino_port) as job:
        job.run()
        job.block_until_disconnected()

@click.command(name="fluorescence_monitoring")
@click.option("--exposure_time", default=10, help="Spectrometer exposure time in seconds.")
@click.option("--interval", default=900, help="Interval between spectrometer measurements in seconds.")
@click.option("--arduino_port", default="/dev/arduino", help="Serial port for Arduino communication.")
def click_fluorescence_monitoring(exposure_time, interval, arduino_port):
    start_fluorescence_monitoring(exposure_time, interval, arduino_port)
# -------------------------------------------------------------------------------------------------


