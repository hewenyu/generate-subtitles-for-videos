# Define variables
PYTHON=python3
PIP=pip3
PYINSTALLER=pyinstaller
MAIN_SCRIPT=${BASE_PATH}/app.py
DIST_DIR=dist
BASE_PATH=$(shell pwd)
# LINUX_BINARY_SPEC=${DIST_DIR}/linux_binary.spec
LINUX_BINARY=${DIST_DIR}/linux_binary
WINDOWS_BINARY=${DIST_DIR}/windows_binary.exe

# Default target
all: install-deps build-linux build-windows

mkdir:
	mkdir -p $(DIST_DIR)

# Install dependencies
install-deps:
	$(PIP) install -r requirements.txt

# Build binary for Linux
build-linux: mkdir
	$(PYINSTALLER) --onefile --name $(LINUX_BINARY) $(MAIN_SCRIPT)

# Build binary for Windows
build-windows: mkdir
	$(PYINSTALLER) --onefile --name $(WINDOWS_BINARY) --windowed $(MAIN_SCRIPT)

# Clean up build artifacts
clean:
	rm -rf build dist *.spec

.PHONY: all install-deps build-linux build-windows clean