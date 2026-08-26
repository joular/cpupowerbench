# <a href="https://www.noureddine.org/research/joular/"><img src="https://raw.githubusercontent.com/joular/.github/main/profile/joular.png" alt="Joular Project" width="64" /></a> CPU Power Benchmark (CPUPowerBench)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue)](https://www.gnu.org/licenses/gpl-3.0)

CPUPowerBench is an automated benchmark to accurately generate a power model for single-board computers (Raspberry Pi, Asus TinkerBoard, BeagleBone, etc.).

## Step 1: CPU Benchmark

### :package: Automatic installation

Run the installer file: ```sh installer.sh``` in cpu-benchmark folder.

It installs the build tools, creates a python virtual environment in ```cpu-benchmark/.venv``` for the CPU load generator, and compiles the CPU cycles program.

### :floppy_disk: Manual Installation

First install dev tools: gcc, cmake, make, python 3, pip and venv: ```sudo apt install python3 python3-pip python3-venv gcc make cmake```

Then install python requirements for CPU Load Generator, in a virtual environment:
```
python3 -m venv .venv
./.venv/bin/pip install -r cpuload/requirements.txt
```

Then compile CPU Cycles program:
```
cmake .
make
```

The compiled program is written to ```cpucycles/cpucycles```.

### :bulb: Usage

Connect your single-board device to a PowerSpy2 power meter.

Then, just run the benchmark script: ```sh start-benchmark.sh```.

The script benchmarks 20 CPU loads, from 5% to 100%, for 60 seconds each, and takes about 25 minutes.
The duration of a load step, the warm up and cool down times, and the list of loads can all be changed at the top of the script.

At the end of the experiment, two CSV files will be generated (cpucycles.csv and cpuload.csv).
Results of a previous run found in the folder are renamed with a timestamp rather than appended to, so two runs are never mixed in the same file.
Finally, get the power meter data by using PowerSpy software and downloading the saved monitored data from the meter's internal memory, and rename the file to powerspy.csv.

### :wrench: Generating a CPU load on its own

The CPU load generator used by the benchmark can also be run by itself, for instance to hold a load while measuring something else:

```
./.venv/bin/python cpuload/cpu_load_generator.py -l 0.5 -d 60
```

- ```-l``` target load per core, between 0 and 1 (default 0.2). One value applies to every core, or one value per core.
- ```-d``` duration in seconds. Negative or omitted runs until the process is interrupted (default -1).
- ```-c``` core(s) to load, defaults to every available core. The benchmark relies on this default to load the whole CPU.
- ```-p``` save a plot of the resulting load as a PNG, only with a fixed duration and a single core.

## Step 2: Power model generation

### :package: Installation

Install python requirements: ```pip install -r requirements.txt``` in model-generation folder.

Python 3.12 or later is required. This step is usually run on a desktop computer rather than on the board itself.

### :bulb: Usage

Copy the 2 CSV files generated in the benchmark (cpucycles.csv and cpuload.csv) to the model-generation folder.
Also, download power data CSV file from PowerSpy2, and copy it under the name powerspy.csv.

Then, just run the model generation script: ```python runModelGeneration.py``` in model-generation folder.

The script works out by itself how the power file is written: whether its columns are separated by a tab, a semicolon or a comma, whether it starts with a header row, and whether it is a PowerSpy2 file (7 columns) or a regular CSV with two columns, timestamp and power consumption.
A timestamp given as a number of seconds since the epoch is read as UTC and moved to the local time of the machine, which is the clock cpucycles.csv is written with.

The older ```python runModelGeneration.py powercsv``` command still works, the option is simply no longer needed.

The script prints a linear and a polynomial power model, and writes two files:

- ```completeData.csv```, the power and CPU utilization samples kept for the model, with the power each model estimates for them.
- ```powerModel.png```, a plot of the measured samples against both models.

The three input files are only read, never modified, so the script can be run again on the same data.

The benchmark only records the time of day, so the date of the experiment is taken from the power file, the only one of the three carrying a full date.
Set the ```year```, ```month``` and ```day``` variables at the top of the script to override it, for a power file that holds a time of day only.

If the clock of the power meter is offset from the clock of the board, set ```CLOCKSYNC``` to the difference in seconds.
The script prints the range covered by each file and the offset between them when they do not overlap at all.

## :bookmark_tabs: Cite this work

To cite our work in a research paper, please cite our paper in Sustainable Computing: Informatics and Systems journal.

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
