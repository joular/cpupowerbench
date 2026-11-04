/*
 * Copyright (c) 2026, Adel Noureddine.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the
 * GNU General Public License v3.0 only (GPL-3.0-only)
 * which accompanies this distribution, and is available at
 * https://www.gnu.org/licenses/gpl-3.0.en.html
 *
 * Author: Adel Noureddine
 */

#ifndef USAGE_H
#define USAGE_H

// Time spent by CPUs since boot, in clock ticks
struct cpu_times {
    unsigned long long busy;
    unsigned long long total;
};

// Reads all the CPUs together into all, and up to max of them one by one into cpus
// Returns the number of CPUs read one by one, or -1 if /proc/stat cannot be read
int read_cpu_times(struct cpu_times *all, struct cpu_times *cpus, int max);

// Share of the time between two readings the CPUs were busy, from 0 to 1
double cpu_usage(const struct cpu_times *before, const struct cpu_times *after);

#endif
