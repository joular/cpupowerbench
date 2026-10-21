# <a href="https://www.noureddine.org/research/joular/"><img src="https://raw.githubusercontent.com/joular/.github/main/profile/joular.png" alt="Joular Project" width="64" /></a> CPU Power Benchmark

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue)](https://www.gnu.org/licenses/gpl-3.0)

CPUPowerBench generates the power model of a single-board computer (Raspberry Pi, Asus Tinker Board, etc.) or of any Linux machine: its power consumption as a function of its CPU utilization.
The CPU of board is stressed step by step while a power meter records its power consumption, then a polynomial model is fitted to the two.
The model can be used by [PowerJoular](https://github.com/joular/powerjoular).

## Step 1: run the benchmark on the board

The benchmark needs Linux and a C compiler. Raspberry Pi OS already has gcc, otherwise install it with `sudo apt install gcc`.

```
sh benchmark/start-benchmark.sh
```

It is compiled on its first run.
The board stays idle for 30 seconds, then runs 20 CPU loads from 5% to 100% of every CPU for 30 seconds each, with 5 idle seconds after each one (takes around 12 minutes in total).
Results are saved to `benchmark-<host>-<date>.csv` in the current folder, so a previous run is never overwritten.

Each CPU gets its own thread, pinned to it, which works for the requested share of every short period and sleeps for the rest.
Periods last 10 ms on average, so a load of 20% keeps every CPU 20% busy all along rather than in bursts at full speed.
Their length varies at random between 5 and 15 ms, otherwise the work would always meet the kernel's accounting tick at the same point and the CPU usage measured would be off by a few points.
Each step prints the CPU usage measured, with the lowest and highest CPU, to check every CPU carries the load.

Check the whole setup in under two minutes first, with fewer and shorter steps:

```
sh benchmark/start-benchmark.sh -w 20 -d 10 -s 25
```

| Option | Default | |
|---|---|---|
| `-w SECONDS` | 30 | idle time before the first load |
| `-d SECONDS` | 30 | time at each load |
| `-c SECONDS` | 5 | idle time after each load |
| `-s PERCENT` | 5 | loads go from this percentage to 100% in steps of it |
| `-l LOADS` | | loads to run instead, between 0 and 1, comma separated |
| `-o FILE` | | output file |

To hold a single load, for instance 50% for a minute: `sh benchmark/start-benchmark.sh -l 0.5 -w 0 -c 0 -d 60`.

For an accurate model:

- Stop other programs. The benchmark warns when the board is busy while it should be idle.
- Keep the clock synchronized (NTP). The benchmark warns when it is not.
- Cool the board. Temperature and frequency are recorded, and throttling is reported in step 3.

## Step 2: record the power

Start recording power consumption before the benchmark and stop after it, with one reading per second. Either:

- Record in the internal memory of a PowerSpy2 meter, then export the data with the PowerSpy software.
- Or run [PowerSpyCli](https://github.com/joular/powerspycli) on another computer: `python powerspycli.py -m 00:11:22:33:44:55 -f power.csv`. Never run it on the measured board, as it would add its own load.

Any other meter works if it exports a time and a power in watts on each row, separated by a semicolon, a comma or a tab.
The time can be a UNIX time, seconds since the epoch (or milliseconds or microseconds), or a date and time, with or without a UTC offset, day or month first.
Decimal commas are fine, readings faster than once a second are averaged over each second, and rows without a time or a power are skipped.
A file with two columns is read as time then power, otherwise give the columns with `--time-column` and `--power-column`.

## Step 3: generate the model

On any computer with Python 3.12 or later, with the libraries in a virtual environment (recent Debian, Raspberry Pi OS and Homebrew refuse to install them system-wide):

```
python3 -m venv .venv
.venv/bin/pip install -r model/requirements.txt
.venv/bin/python model/generate_model.py power.csv benchmark-raspberrypi-20261020-140312.csv
```

The script lines up the clocks of the meter and the board by itself, and drops the first and last 2 seconds of each load.
The power at rest comes from the warmup, without its first 10 seconds: the ondemand governor of Raspberry Pi OS keeps the full frequency for 5 seconds after a busy moment, which also rules out the short idle steps between loads.
Only polynomials that never decrease between 0% and 100% are fitted, and the degree is the lowest that predicts each load.

| Option | |
|---|---|
| `--offset SECONDS` | seconds the meter's clock is ahead of the board's, instead of finding it |
| `--trim SECONDS` | seconds dropped at both ends of each load (default 2) |
| `--max-degree N` | highest degree tried (default 9) |
| `--time-column N` | column of the power file holding the time (default 1) |
| `--power-column N` | column of the power file holding the power (default 2, or 4 for a PowerSpy2 export) |

It prints a linear and a polynomial model, the coefficients ready for PowerJoular, and writes next to the benchmark file:

- `<name>-data.csv`, the samples kept, with the power each model estimates for them.
- `<name>-model.json`, both models and their errors.
- `<name>-model.png`, the measured samples against both models, and the power and CPU utilization over time.

## Benchmark file

A first line describes the board (`# model, cores, governor, kernel`), then one row per second, separated by semicolons:

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

## :newspaper: License

CPU Power Benchmark is licensed under the GNU GPL 3 license only (GPL-3.0-only).

Copyright (c) 2026, Adel Noureddine.
All rights reserved. This program and the accompanying materials are made available under the terms of the GNU General Public License v3.0 only (GPL-3.0-only) which accompanies this distribution, and is available at: https://www.gnu.org/licenses/gpl-3.0.en.html

Author: Adel Noureddine
