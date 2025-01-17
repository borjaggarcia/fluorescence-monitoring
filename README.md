# fluorescence-monitoring
**Fluorescence monitoring plugin for Pioreactor systems: interfaces with Arduino for excitation LED control and automates spectrometer measurements.**

## How to Install the Plugin

1. Ensure you have the latest version of Python and pip:

   ```bash
   sudo apt install python3 python3-pip
   ```

2. To make the spectrometer work, you need libusb, pyusb, and rgbdriverkit.

   2.1. Install libusb:
   ```bash
   sudo apt install libusb-1.0-0-dev
   ```

   2.2. Install pyusb:
   ```bash
   pip3 install pyusb
   ```

   2.3. Install matplotlib (for graphs):
   ```bash
   sudo apt install python3-matplotlib
   ```

   2.4. Install the spectrometer driver:
   - Copy the folder `pyrgbdriverkit-0.3.7` to the Pioreactor.
   - Navigate to the folder containing `setup.py` and execute:
   ```bash
   sudo pip3 install .
   ```

3. Create udev rules.

   **For the spectrometer:**
   ```bash
   sudo nano /etc/udev/rules.d/99-spectrometer.rules
   ```
   Add the following line:
   ```
   SUBSYSTEM=="usb", ATTRS{idVendor}=="276e", ATTRS{idProduct}=="0208", MODE="0666"
   ```

   **For the Arduino:**
   ```bash
   sudo nano /etc/udev/rules.d/99-arduino.rules
   ```
   Add the following line:
   ```
   SUBSYSTEM=="tty", ATTRS{serial}=="393D99A6EA0D4AE8", SYMLINK+="arduino"
   ```

   Save the file and reload udev rules:
   ```bash
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   ```
   Disconnect and reconnect the devices to apply the new rules.

4. Place the `fluorescence_monitoring.py` file in the plugins folder.

5. Add the corresponding `.yaml` files in `ui/Jobs`, `ui/Charts`, and/or `ui/Automations` as needed.

6. Modify the `config.ini` file for charts:

   Add the following section:
   ```ini
   [ui.overview.charts]
   # show/hide charts on the PioreactorUI dashboard
   # 1 is show, 0 is hide
   implied_growth_rate=1
   implied_daily_growth_rate=0
   fraction_of_volume_that_is_alternative_media=0
   normalized_optical_density=1
   raw_optical_density=1
   temperature=1
   intensity_700nm_excitation_655nm_chart=1
   intensity_658nm_excitation_655nm_chart=1
   intensity_700nm_excitation_620nm_chart=1
   intensity_658nm_excitation_620nm_chart=1
   intensity_700nm_excitation_595nm_chart=1
   intensity_658nm_excitation_595nm_chart=1
   intensity_700nm_excitation_527nm_chart=1
   intensity_658nm_excitation_527nm_chart=1
   intensity_700nm_excitation_450nm_chart=1
   intensity_658nm_excitation_450nm_chart=1
   intensity_700nm_excitation_405nm_chart=1
   intensity_658nm_excitation_405nm_chart=1


