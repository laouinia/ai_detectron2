PYTHON = py -3.10

# Targets:
main: detector_main.py detector.py
	$(PYTHON) detector_main.py

device: detector_device.py
	$(PYTHON) detector_device.py


plotter: detector_plotter.py
	$(PYTHON) detector_plotter.py

train: detector_train.py
	$(PYTHON) detector_train.py

test: detector_test.py
	$(PYTHON) detector_test.py
lint:
	$(PYTHON) -m pylint detector_main.py detector.py detector_plotter.py

clean:
	rd /s /q __pycache__ && rd /s /q .mypy_cache
