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

if [ ! -x "$DIR/cpupowerbench" ] || [ -n "$(cd "$DIR" && find main.c usage.c usage.h load.c load.h -newer cpupowerbench)" ]; then
	if ! command -v cc >/dev/null 2>&1; then
		echo "A C compiler is needed, install it with: sudo apt install gcc" >&2
		exit 1
	fi
	echo "Compiling cpupowerbench"
	(cd "$DIR" && cc -O2 -Wall -Wextra -pthread -o cpupowerbench main.c usage.c load.c)
fi

exec "$DIR/cpupowerbench" "$@"
