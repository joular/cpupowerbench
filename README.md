# <a href="https://www.noureddine.org/research/joular/"><img src="https://raw.githubusercontent.com/joular/.github/main/profile/joular.png" alt="Joular Project" width="64" /></a> CPU Power Benchmark

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue)](https://www.gnu.org/licenses/gpl-3.0)

CPUPowerBench generates the power model of a single-board computer (Raspberry Pi, Asus Tinker Board, etc.) or of any Linux machine: its power consumption as a function of its CPU utilization.
The CPU of the board is stressed step by step while a power meter records its power consumption, then a polynomial model is fitted to the two.
The model gives the power of the whole board as drawn at the wall, power supply and peripherals included, not that of the CPU alone.
The models of single-board computers are built into [joularcore](https://github.com/joular/joularcore), which [PowerJoular](https://github.com/joular/powerjoular) uses, and published in the [Power Models Database](https://github.com/joular/powermodels).

## Step 1: run the benchmark on the board

The benchmark needs Linux, a C compiler and make. Raspberry Pi OS already has them, otherwise install them with `sudo apt install gcc make`.

```
sh benchmark/start-benchmark.sh
```

It is compiled on its first run, then waits 60 seconds for the board to cool down and the power meter to be started.
The board stays idle for 30 seconds, then runs 20 CPU loads from 5% to 100% of every CPU for 30 seconds each, with 5 idle seconds after each one (takes around 13 minutes in total).
Results are saved to `benchmark-<host>-<date>.csv` in the current folder, so a previous run is never overwritten.
Over SSH, run it in `tmux` or as `nohup sh benchmark/start-benchmark.sh &`, so that a dropped connection does not stop it.

Each CPU gets its own thread, pinned to it, which works for the requested share of every short period and sleeps for the rest.
Periods last 10 ms on average, so a load of 20% keeps every CPU 20% busy all along rather than in bursts at full speed.
Their length varies at random between 5 and 15 ms, otherwise the work would always meet the kernel's accounting tick at the same point and the CPU usage measured would be off by a few points.
Each step prints the CPU usage measured, with the lowest and highest CPU, to check every CPU carries the load.

Check the whole setup in under three minutes first, with fewer and shorter steps:

```
sh benchmark/start-benchmark.sh -w 20 -d 10 -s 25
```

| Option | Default | |
|---|---|---|
| `-w SECONDS` | 30 | idle time before the first load |
| `-d SECONDS` | 30 | time at each load |
| `-c SECONDS` | 5 | idle time after each load |
| `-s PERCENT` | 5 | loads go from this percentage to 100% in steps of it |
| `-l LOADS` | | loads to run instead of `-s`, from 0.001 to 1, comma separated |
| `-o FILE` | | output file |

To hold a single load, for instance 50% for a minute: `sh benchmark/start-benchmark.sh -l 0.5 -w 0 -c 0 -d 60`.

For an accurate model:

- Stop other programs. The benchmark warns when the board is busy while it should be idle.
- Keep the clock synchronized (NTP). The benchmark warns when it is not.
- Cool the board and power it well. Temperature and frequency are recorded, as are the firmware's under-voltage and throttling flags on a Raspberry Pi, and throttling is reported in step 3.

## Step 2: record the power

Start recording power consumption before the benchmark and stop after it, with one reading per second. Either:

- Record in the internal memory of a PowerSpy2 meter, then export the data with the PowerSpy software.
- Or run [PowerSpyCli](https://github.com/joular/powerspycli) on another computer: `python powerspycli.py -m 00:11:22:33:44:55 -f power.csv`. Never run it on the measured board, as it would add its own load.

Any other meter works if it exports a time and a power in watts on each row, separated by a semicolon, a comma or a tab.
The time can be a UNIX time, seconds since the epoch (or milliseconds or microseconds), or a date and time, with or without a UTC offset, day or month first.
A date without a UTC offset is read as UTC, unless its time zone is given with `--tz`: the PowerSpy software exports in the local time of its computer.
Decimal commas are fine, readings faster than once a second are averaged over each second, and rows without a time or a power are skipped.
A file with two columns is read as time then power, otherwise give the columns with `--time-column` and `--power-column`.

## Step 3: generate the model

On any computer with Python 3.12 or later, with the libraries in a virtual environment (recent Debian, Raspberry Pi OS and Homebrew refuse to install them system-wide):

```
python3 -m venv .venv
.venv/bin/pip install -r model/requirements.txt
.venv/bin/python model/generate_model.py power.csv benchmark-raspberrypi-20261020-140312.csv
```

The script lines up the clocks of the meter and the board by itself, searching up to 120 seconds each way around `--offset`, and drops the first and last 2 seconds of each load.
The power at rest comes from the warmup, without its first 10 seconds: the ondemand governor of Raspberry Pi OS keeps the full frequency for 5 seconds after a busy moment, which also rules out the short idle steps between loads.
Only polynomials that never decrease between 0% and 100% are fitted: their coefficients in Bernstein form are kept non-decreasing, which guarantees the polynomial never decreases, though a few polynomials that never decrease are out of its reach.
The degree is chosen by cross-validation: every load step is predicted by a fit on all the other steps, except the first and last steps, which would have to be extrapolated.
The degree kept is the lowest whose error is within 2% of the best, among degrees up to the number of load levels minus 2.

| Option | |
|---|---|
| `--offset SECONDS` | seconds the meter's clock is ahead of the board's, the middle of the search (default 0) |
| `--no-search` | take `--offset` as it is |
| `--tz ZONE` | time zone of the dates without a UTC offset in the power file, such as `Europe/Paris` (default UTC) |
| `--trim SECONDS` | seconds dropped at both ends of each load (default 2) |
| `--max-degree N` | highest degree tried (default 9) |
| `--time-column N` | column of the power file holding the time (default 1) |
| `--power-column N` | column of the power file holding the power (default 2, or 4 for a PowerSpy2 export) |
| `--codename NAME` | name of the model in the JSON file, as in the Power Models Database (default: the board name, with `-64` on a 64-bit system) |

It prints a linear and a polynomial model, the coefficients ready for joularcore, and writes next to the benchmark file, never over an existing file:

- `<name>-data.csv`, the samples kept, with the power each model estimates for them.
- `<name>-model.json`, both models and their errors, laid out as in the Power Models Database.
- `<name>-model.png`, the measured samples against both models, and the power and CPU utilization over time.

## Adding a model to joularcore

joularcore holds the models of single-board computers in `src/linux/joular_core-rpi.adb`, and picks one from the board name in `/proc/device-tree/model`. To add a board:

1. Add it to the `Model_Name` enumeration, named like the others. A model measured on a 64-bit system ends in `_64` there and in `-64` in the Power Models Database: the 32-bit and 64-bit systems of a board draw different powers, and only the model of the system measured is used.
2. Add its coefficients to the `Models` aggregate: the ten numbers of the `joularcore:` line printed in step 3.
3. Add a branch to `Model_Of` matching its board name, before any more general match: "Raspberry Pi 4 Model B Rev 1.1" must come before "Raspberry Pi 4 Model B", which matches it too.

joularcore uses a model only when the machine has no RAPL counters, and only for the board names it knows from the device tree: the model of any other Linux machine is not loaded automatically, it is for your own use.
Please also share your model in the [Power Models Database](https://github.com/joular/powermodels).

## Benchmark file

A first line describes the board (`# model, cores, governor, kernel, bits` and, on a Raspberry Pi, the firmware's throttling flags since boot in hexadecimal), then one row per second, separated by semicolons:

| Column | |
|---|---|
| `time` | end of the second, in seconds since the epoch |
| `usage` | CPU utilization measured over that second, from 0 to 1 |
| `load` | load the benchmark was running, 0 when idle |
| `step` | number of the step, starting at 0 |
| `temperature` | in °C, empty if the board does not report it |
| `frequency` | of the first CPU, in MHz, empty if the board does not report it |

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

CPU Power Benchmark is licensed under the GNU GPL 3 license only (GPL-3.0-only).

Copyright (c) 2026, Adel Noureddine.
All rights reserved. This program and the accompanying materials are made available under the terms of the GNU General Public License v3.0 only (GPL-3.0-only) which accompanies this distribution, and is available at: https://www.gnu.org/licenses/gpl-3.0.en.html

Author: Adel Noureddine
