#!/bin/sh

# Copyright (c) 2026, Adel Noureddine.
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the
# GNU General Public License v3.0 only (GPL-3.0-only)
# which accompanies this distribution, and is available at
# https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Author: Adel Noureddine

# Compiles the benchmark when needed, then runs it with the given options.
# Results are written in the current folder.

set -e

DIR=$(cd "$(dirname "$0")" && pwd)

if ! command -v cc >/dev/null 2>&1 || ! command -v make >/dev/null 2>&1; then
	echo "A C compiler and make are needed, install them with: sudo apt install gcc make" >&2
	exit 1
fi
(cd "$DIR" && make)

if [ "$1" != "-h" ]; then
	# Lets the board cool down after compiling, and leaves time to start the power meter
	echo "Starting in 60 s, start recording the power now"
	sleep 60
fi

exec "$DIR/cpupowerbench" "$@"
