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

#include <stdio.h>
#include <string.h>

#include "usage.h"

static int parse_times(const char *line, struct cpu_times *times) {
    unsigned long long user = 0, nice = 0, system = 0, idle = 0;
    unsigned long long iowait = 0, irq = 0, softirq = 0, steal = 0;

    // Older kernels stop before steal, which then stays 0. guest and guest_nice come after it and are left out, as user and nice already hold them. Same sum as PowerJoular's
    if (sscanf(line, "%*s %llu %llu %llu %llu %llu %llu %llu %llu",
               &user, &nice, &system, &idle, &iowait, &irq, &softirq, &steal) < 4) {
        return -1;
    }

    times->busy = user + nice + system + irq + softirq + steal;
    times->total = times->busy + idle + iowait;
    return 0;
}

int read_cpu_times(struct cpu_times *all, struct cpu_times *cpus, int max) {
    char line[512];
    int count = 0;

    FILE *file = fopen("/proc/stat", "r");
    if (file == NULL) {
        return -1;
    }

    // The first line adds up every CPU, then come cpu0, cpu1...
    if (fgets(line, sizeof(line), file) == NULL || strncmp(line, "cpu ", 4) != 0
        || parse_times(line, all) != 0) {
        fclose(file);
        return -1;
    }
    while (count < max && fgets(line, sizeof(line), file) != NULL && strncmp(line, "cpu", 3) == 0) {
        if (parse_times(line, &cpus[count]) == 0) {
            count++;
        }
    }

    fclose(file);
    return count;
}

double cpu_usage(const struct cpu_times *before, const struct cpu_times *after) {
    // iowait may go backwards on some kernels, so the total is not always increasing
    if (after->total <= before->total || after->busy < before->busy) {
        return 0.0;
    }

    double usage = (double) (after->busy - before->busy) / (double) (after->total - before->total);
    return usage > 1.0 ? 1.0 : usage;
}
