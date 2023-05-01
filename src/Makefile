PYTHON = py -3.10

# Targets:
all: detector_main.py detector.py detector_plotter.py
	$(PYTHON) detector_main.py

plotter: detector_plotter.py
	$(PYTHON) detector_plotter.py

train: detector_train.py
	$(PYTHON) detector_train.py

test: detector_test.py
	$(PYTHON) detector_test.py
lint:
	$(PYTHON) -m pylint detector_main.py detector.py detector_plotter.py

clean:
	rd /s /q __pycache__
