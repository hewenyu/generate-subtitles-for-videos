# Define variables
PYTHON=python3
PIP=pip3

# Define the target for building the project with Nuitka
build:
	$(PYTHON) -m nuitka --onefile main.py