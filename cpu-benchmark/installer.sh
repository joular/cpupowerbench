#!/bin/sh

# Copyright (c) 2020-2023, Université de Pays et des Pays de l'Adour.
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the
# GNU General Public License v3.0 only (GPL-3.0-only)
# which accompanies this distribution, and is available at
# https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Author: Houssam Kanso
#
# Contributors: Adel Noureddine

set -e

# Run from the folder holding this script, so the relative paths below hold
# whichever folder the script is called from
cd "$(dirname "$0")"

echo "Installing the build tools and python"
sudo apt update
sudo apt install -y python3 python3-pip python3-venv gcc make cmake

# Debian 12 and the Raspberry Pi OS releases based on it refuse to let pip install
# in the system python (PEP 668), so the dependencies of the CPU load generator go
# in a virtual environment next to this script, which start-benchmark.sh picks up
echo "Creating the python virtual environment in .venv"
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r cpuload/requirements.txt

echo "Compiling the CPU cycles program"
# A previous version of this installer built in the source folder, and cmake refuses to
# build elsewhere while the files of that build are still around
rm -rf cpucycles/CMakeCache.txt cpucycles/CMakeFiles cpucycles/cmake_install.cmake cpucycles/Makefile
cmake -S cpucycles -B cpucycles/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpucycles/build

echo "Installation finished"
