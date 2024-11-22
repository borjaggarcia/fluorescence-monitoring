# fluorescence-monitoring
**Fluorescence monitoring plugin for Pioreactor systems: interfaces with Arduino for excitation LED control and automates spectrometer measurements.**

## Cómo Instalar en Pluging
1. Asegurar de que tienes la última versión de Python y pip:

	sudo apt install python3 python3-pip


2. Para hacer funcionar el espectrómetro, se necesitan libusb, pyusb, y rgbdriverkit.

	2.1. Instalar libusb:
    		sudo apt install libusb-1.0-0-dev

    
	2.2. Instalar pyusb:
    		pip3 install pyusb
  
    
	2.3. Instalar matplotlib (para las gráficas):
    		sudo apt install python3-matplotlib
   
    
	2.4. Instalar el driver del espectrómetro:
    		Copiar la carpeta pyrgbdriverkit-0.3.7 en el pio.
		Colocarse en la misma carpeta en el que se encuentra el setup.py y ejecutar
		sudo pip3 install .

3. Crear reglas udev.

	Espectrometro:

	sudo nano /etc/udev/rules.d/99-spectrometer.rules
	SUBSYSTEM=="usb", ATTRS{idVendor}=="276e", ATTRS{idProduct}=="0208", MODE="0666"

	Arduino:

	sudo nano /etc/udev/rules.d/99-arduino.rules
	SUBSYSTEM=="tty", ATTRS{serial}=="393D99A6EA0D4AE8", SYMLINK+="arduino"


	guardar archivo y recargar las reglas udev:
	sudo udevadm control --reload-rules
	sudo udevadm trigge

	desconectar y conectar para que las nuevas reglas se apliquen


4. Poner el fluorescence_monitoring.py en la carpeta de plugins.

5. En ui/Jobs, Charts y/o Automations poner los .yaml correspondientes.

6. Para los charts modificar el config.ini:

	[ui.overview.charts]
	\# show/hide charts on the PioreactorUI dashboard
	\# 1 is show, 0 is hide
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

