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

sudo apt install python3 python3-pip gcc make cmake
cd cpuload
pip install -r requirements.txt
cd ..
cd cpucycles
cmake .
make
cd ..