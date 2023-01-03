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

./cpucycles/cpucycles &
CPUCYCLES_PID=$!
echo "Starting CPU cycles collecting with PID: $CPUCYCLES_PID"

echo "Sleeping for 60 seconds to warm up"
sleep 60
DURATION=60
for CPU_LOAD in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1;
do
	echo "Benchmarking CPU at load $CPU_LOAD for 60 seconds"
	START_TIME=`date +"%T"`
	python3 ./cpuload/CPULoadGenerator.py -l $CPU_LOAD -d $DURATION
	END_TIME=`date +"%T"`
	echo "$START_TIME,$END_TIME,$CPU_LOAD" >> "cpuload.csv"
    echo "Sleeping 10 seconds to cool down and separate each load benchmark"
	sleep 10
done

kill -KILL $CPUCYCLES_PID

echo "Benchmarking finished"