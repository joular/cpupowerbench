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

# Program collecting the CPU cycles
CPUCYCLES=./cpucycles/cpucycles

# CPU cycles collected during the whole benchmark, written by the program above
CYCLES_CSV=cpucycles.csv

# Start and end time of each load step, written by this script
LOAD_CSV=cpuload.csv

# Duration in seconds of each load step
DURATION=60

# Seconds spent idle before the first load step, so the board starts from a steady state
WARMUP=60

# Seconds spent idle between two load steps, to cool down and separate them
COOLDOWN=10

# CPU loads to benchmark, as a fraction of the whole CPU
LOADS="0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1"

# Python of the virtual environment created by installer.sh, or the system one if there is none
if [ -x ./.venv/bin/python ]; then
	PYTHON=./.venv/bin/python
else
	PYTHON=python3
fi

# Check the dependencies before starting: the benchmark takes about 25 minutes,
# it should not be found at the end of it that nothing was collected
if [ ! -x "$CPUCYCLES" ]; then
	echo "Error: $CPUCYCLES is missing or not executable. Run installer.sh first." >&2
	exit 1
fi
if ! "$PYTHON" -c "import psutil" >/dev/null 2>&1; then
	echo "Error: the psutil python module is missing. Run installer.sh first." >&2
	exit 1
fi
if [ ! -f ./cpuload/cpu_load_generator.py ]; then
	echo "Error: ./cpuload/cpu_load_generator.py is missing." >&2
	exit 1
fi

# Keep the files of a previous run instead of appending this run to them,
# as mixing two runs in one file makes the generated power model wrong
STAMP=`date +"%Y%m%d-%H%M%S"`
for RESULT_FILE in "$CYCLES_CSV" "$LOAD_CSV"; do
	if [ -e "$RESULT_FILE" ]; then
		echo "Moving the $RESULT_FILE of a previous run to $RESULT_FILE.$STAMP"
		mv "$RESULT_FILE" "$RESULT_FILE.$STAMP"
	fi
done

CPUCYCLES_PID=""

# Stop collecting cycles whatever happens, so no collector is left running in the
# background when the benchmark is interrupted
cleanup() {
	if [ -n "$CPUCYCLES_PID" ]; then
		kill -TERM "$CPUCYCLES_PID" 2>/dev/null || true
		wait "$CPUCYCLES_PID" 2>/dev/null || true
		CPUCYCLES_PID=""
	fi
}
trap 'cleanup; echo "Benchmarking interrupted" >&2; exit 130' INT TERM
trap cleanup EXIT

"$CPUCYCLES" &
CPUCYCLES_PID=$!
echo "Starting CPU cycles collecting with PID: $CPUCYCLES_PID"

echo "Sleeping for $WARMUP seconds to warm up"
sleep "$WARMUP"

for CPU_LOAD in $LOADS;
do
	echo "Benchmarking CPU at load $CPU_LOAD for $DURATION seconds"
	START_TIME=`date +"%T"`
	"$PYTHON" ./cpuload/cpu_load_generator.py -l "$CPU_LOAD" -d "$DURATION"
	END_TIME=`date +"%T"`
	echo "$START_TIME,$END_TIME,$CPU_LOAD" >> "$LOAD_CSV"
	echo "Sleeping $COOLDOWN seconds to cool down and separate each load benchmark"
	sleep "$COOLDOWN"
done

cleanup

echo "Benchmarking finished"
echo "Results written to $CYCLES_CSV and $LOAD_CSV"
