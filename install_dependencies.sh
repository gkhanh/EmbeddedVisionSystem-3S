#!/bin/bash

# This script installs all the dependencies required for the vision system program.

echo "Updating package list..."
apt-get update

echo "Installing Python3 and pip..."
apt-get install python3 python3-pip -y

echo "Installing required Python libraries..."
pip3 install opencv-python numpy
pip install scikit-learn matplotlib opencv-python numpy

echo "All dependencies installed successfully."