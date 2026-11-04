# <a href="https://www.noureddine.org/research/joular/"><img src="https://raw.githubusercontent.com/joular/.github/main/profile/joular.png" alt="Joular Project" width="64" /></a> CPU Power Benchmark

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue)](https://www.gnu.org/licenses/gpl-3.0)

CPUPowerBench builds the power model of a single-board computer (Raspberry Pi, Asus Tinker Board, etc.) or of any Linux machine: how much power it draws at each level of CPU usage.

It loads the CPU step by step, from idle to 100%, while a power meter records the power of the board. A polynomial is then fitted to the CPU usage and the power.
The model gives the power of the whole board, as measured at the wall: power supply and peripherals included.

The models of single-board computers are used by [joularcore](https://github.com/joular/joularcore) and [PowerJoular](https://github.com/joular/powerjoular), and shared in the [Power Models Database](https://github.com/joular/powermodels).

You need:

- The board, running Linux, with a C compiler and make. Raspberry Pi OS has them, otherwise install them with `sudo apt install gcc make`.
- A power meter that records one reading per second, such as a PowerSpy2.
- A computer with Python 3.12 or later, to build the model.

## Step 1: run the benchmark on the board

```
sh benchmark/start-benchmark.sh
```

The benchmark:

1. Compiles itself on its first run.
2. Waits 60 seconds, to let you start the power meter and the board cool down.
3. Stays idle for 30 seconds.
4. Runs 20 loads, from 5% to 100% in steps of 5%, for 30 seconds each, with 5 idle seconds after each one.

It takes about 13 minutes. Results are saved to `benchmark-<host>-<date>.csv` in the current folder, so a new run never overwrites an old one.
Ctrl+C stops the benchmark and keeps what was measured.
Over SSH, run it in `tmux` or with `nohup sh benchmark/start-benchmark.sh &`, so that a dropped connection does not stop it.

To check your setup first, this shorter run takes less than 3 minutes:

```
sh benchmark/start-benchmark.sh -w 20 -d 10 -s 25
```

| Option | Default | |
|---|---|---|
| `-p SECONDS` | 60 | wait before starting |
| `-w SECONDS` | 30 | idle time before the first load |
| `-d SECONDS` | 30 | time at each load |
| `-c SECONDS` | 5 | idle time after each load |
| `-s PERCENT` | 5 | loads go from this percentage to 100%, in steps of it |
| `-l LOADS` | | your own loads instead of `-s`, from 0.001 to 1, separated by commas (e.g. `0.25,0.5,1`) |
| `-o FILE` | | output file |

For example, to hold 50% for one minute: `sh benchmark/start-benchmark.sh -l 0.5 -w 0 -c 0 -d 60`.

For an accurate model:

- Stop other programs. The benchmark warns you if the board is busy when it should be idle.
- Keep the clock in sync with NTP. The benchmark warns you if it is not.
- Use a good power supply and cooling. On a Raspberry Pi, under-voltage and throttling are shown after each step, and reported again in step 3.

### How the load works

Each CPU gets its own thread, which works for a share of every short period and sleeps for the rest. At 20%, it works 2 ms out of every 10 ms, so the CPU stays at 20% for the whole step. The work mixes integer, memory and floating point operations, like normal programs.
The periods last between 5 and 15 ms, at random, so that the kernel's CPU accounting does not miss part of the work.

After each step, the benchmark prints the CPU usage it measured, and the lowest and highest CPU, so you can check that every CPU carries the load.

## Step 2: record the power

Start recording before the benchmark starts (during its 60 seconds wait) and stop after it ends, with one reading per second.

With a PowerSpy2 meter, either:

- Record in the meter's memory, then export the data with the PowerSpy software.
- Or run [PowerSpyCli](https://github.com/joular/powerspycli) on another computer: `python powerspycli.py -m 00:11:22:33:44:55 -f power.csv`. Do not run it on the board you measure, as it would add its own load.

Any other meter works if it saves a file with a time and a power in watts on each row:

- Columns are separated by a semicolon, a comma or a tab. Decimal commas are fine.
- The time is a UNIX time (in seconds, milliseconds or microseconds), or a date. A date without a time zone is read as UTC, or in the zone given with `--tz` in step 3.
- With two columns, the first is the time and the second the power. Otherwise, give their numbers with `--time-column` and `--power-column`.
- Several readings in the same second are averaged, and rows without a time or a power are skipped.

## Step 3: generate the model

On any computer with Python 3.12 or later:

```
python3 -m venv .venv
.venv/bin/pip install -r model/requirements.txt
.venv/bin/python model/generate_model.py power.csv benchmark-raspberrypi-20261020-140312.csv
```

The virtual environment is needed on recent Debian, Raspberry Pi OS and Homebrew, which refuse to install Python libraries for the whole system.

The script:

- Finds the offset between the clocks of the meter and of the board by itself.
- Drops the first and last 2 seconds of each load, where the power is still changing.
- Takes the idle power from the first idle period only, without its first 10 seconds. The ondemand governor of Raspberry Pi OS keeps the CPU at full speed for 5 seconds after any work, so the short idle periods between loads read too high.
- Fits polynomials that never go down: the power never drops when the CPU works more.
- Keeps the lowest degree whose error in cross-validation is within 2% of the best one, up to degree 9.

The model uses the CPU usage measured each second, not the load the benchmark asked for.

| Option | |
|---|---|
| `--offset SECONDS` | seconds the meter's clock is ahead of the board's, used as is instead of searching |
| `--tz ZONE` | time zone of dates written without one, such as `Europe/Paris` (default UTC) |
| `--trim SECONDS` | seconds dropped at both ends of each load (default 2) |
| `--max-degree N` | highest degree tried (default 9) |
| `--time-column N` | column of the time in the power file (default 1) |
| `--power-column N` | column of the power in the power file (default 2, or 4 for a PowerSpy2 export) |
| `--codename NAME` | name of the model in the JSON file (default: the board name, ending in `-64` on a 64-bit system) |

It prints a linear model, a polynomial model and the coefficients for joularcore. It also writes three files next to the benchmark file, never over an existing one:

- `<name>-data.csv`: the samples used, with the power each model gives for them.
- `<name>-model.json`: both models and their errors, in the format of the Power Models Database.
- `<name>-model.png`: the samples against both models, and the power and CPU usage over time.

## Add a model to joularcore

joularcore keeps the models of single-board computers in `src/linux/joular_core-rpi.adb`, and picks one from the board name in `/proc/device-tree/model`. To add a board:

1. Add it to the `Model_Name` list, named like the others. A model measured on a 64-bit system ends in `_64` (and in `-64` in the Power Models Database): the same board draws a different power on 32 and 64 bits, so joularcore uses a `_64` model on 64-bit systems only.
2. Add its coefficients to `Models`: the ten numbers of the `joularcore:` line printed in step 3.
3. Add its board name to `Model_Of`, before any shorter name that also matches it: "Raspberry Pi 4 Model B Rev 1.1" must come before "Raspberry Pi 4 Model B".

joularcore uses these models only on machines without RAPL, and only for the board names it knows. The model of any other Linux machine is for your own use.

Please also share your model in the [Power Models Database](https://github.com/joular/powermodels).

## Benchmark file

The first line describes the board: `# model, cores, governor, kernel, bits`. Then comes one row per second, with columns separated by semicolons:

| Column | |
|---|---|
| `time` | end of the second, in seconds since 1970 (UNIX time) |
| `usage` | CPU usage measured over that second, from 0 to 1 |
| `load` | load the benchmark was running, 0 when idle |
| `step` | step number, from 0 |
| `temperature` | in °C, empty if the board does not report it |
| `frequency` | highest frequency of the CPUs, in MHz, empty if the board does not report it |
| `throttled` | Raspberry Pi firmware flags, in hexadecimal as `vcgencmd get_throttled` shows them, empty on other boards |

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
