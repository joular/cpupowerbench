# <a href="https://www.noureddine.org/research/joular/"><img src="https://raw.githubusercontent.com/joular/.github/main/profile/joular.png" alt="Joular Project" width="64" /></a> CPU Power Benchmark (CPUPowerBench)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue)](https://www.gnu.org/licenses/gpl-3.0)

CPUPowerBench is an automated benchmark to accurately generate a power model for single-board computers (Raspberry Pi, Asus TinkerBoard, BeagleBone, etc.).

## Step 1: CPU Benchmark

### :package: Automatic installation

Run the installer file: ```sh installer.sh``` in cpu-benchmark folder.

### :floppy_disk: Manual Installation

First install dev tools: gcc, cmake, make, python 3, pip: ```sudo apt install python3 python3-pip gcc make cmake```

Then install python requirements for CPU Load Generator: ```pip install -r requirements.txt```

Then compile CPU Cycles program:
```
cmake .
make
```

### :bulb: Usage

Connect your single-board device to a PowerSpy2 power meter.

Then, just run the benchmark script: ```sh start-benchmark.sh```.

At the end of the experiment, two CSV files will be generated (cpucycles.csv and cpuload.csv).
Finally, get the power meter data by using PowerSpy software and downloading the saved monitored data from the meter's internal memory, and rename the file to powerspy.csv.

## Step 2: Power model generation

### :package: Installation

Install python requirements: ```pip install -r requirements.txt``` in model-generation folder.

### :bulb: Usage

Copy the 2 CSV files generated in the benchmark (cpucycles.csv and cpuload.csv) to the model-generation folder.
Also, download power data CSV file from PowerSpy2, and copy it under the name powerspy.csv.

Then, just run the model generation script: ```python runModelGeneration.py``` in model-generation folder.

Note: if you use a different powermeter that outputs a different CSV file structure, you need to modify the model generation file to process the CSV file accordingly.

## :bookmark_tabs: Cite this work

To cite our work in a research paper, please cite our paper in the 18th International Conference on Intelligent Environments (IE2022).

- **Automated Power Modeling of Computing Devices: Implementation and Use Case for Raspberry Pis**. Houssam Kanso, Adel Noureddine, and Ernesto Exposito. In Sustainable Computing: Informatics and Systems journal (SUSCOM). Volume 37. January 2023.

```
@article{KANSO2023100837,
	title = {Automated power modeling of computing devices: Implementation and use case for Raspberry Pis},
	journal = {Sustainable Computing: Informatics and Systems},
	volume = {37},
	pages = {100837},
	year = {2023},
	issn = {2210-5379},
	doi = {https://doi.org/10.1016/j.suscom.2022.100837},
	url = {https://www.sciencedirect.com/science/article/pii/S2210537922001688},
	author = {Houssam Kanso and Adel Noureddine and Ernesto Exposito},
	keywords = {Power consumption, Performance, Measurement, Empirical experimentation, Automated software architecture}
}
```

## :newspaper: License

RPiPowerBench is licensed under the GNU GPL 3 license only (GPL-3.0-only).

Copyright (c) 2020-2023, Université de Pau et des Pays de l'Adour.
All rights reserved. This program and the accompanying materials are made available under the terms of the GNU General Public License v3.0 only (GPL-3.0-only) which accompanies this distribution, and is available at: https://www.gnu.org/licenses/gpl-3.0.en.html

Authors : Houssam Kanso, Adel Noureddine
