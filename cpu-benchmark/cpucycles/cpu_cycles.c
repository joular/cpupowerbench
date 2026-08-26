/* Copyright (c) 2020-2023, Université de Pays et des Pays de l'Adour.
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the
# GNU General Public License v3.0 only (GPL-3.0-only)
# which accompanies this distribution, and is available at
# https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Author : Adel Noureddine
*/

#include <stdio.h>
#include <string.h>

#include "cpu_cycles.h"

#define PROC_STAT_FILE "/proc/stat"

int calculate_cpu_cycles(struct cpu_cycles_info * cpu_data) {
    // Start from a clean snapshot, so a partial read never leaves stale values behind
    memset(cpu_data, 0, sizeof(*cpu_data));

    // Open /proc/stat file
    FILE * fp;
    if ((fp = fopen(PROC_STAT_FILE, "r")) == NULL) {
        fprintf(stderr, "Error in reading file %s.\n", PROC_STAT_FILE);
        return -1;
    }

    // Reading cpu cycles from the first line of /proc/stat
    // Example line is: cpu  83141 56 28074 2909632 3452 10196 3416 0 0 0
    // which is the time spent in user mode, in user mode at a low priority (nice), in system mode,
    // in the idle task, waiting for I/O, handling interrupts, handling soft interrupts,
    // and stolen by the hypervisor of a virtual machine
    char line[256];
    if (fgets(line, sizeof(line), fp) == NULL) {
        fprintf(stderr, "Error in reading file %s.\n", PROC_STAT_FILE);
        fclose(fp);
        return -1;
    }
    fclose(fp);

    // The line must be the one totalling every core (so "cpu ") rather than the one of a single core ("cpu0")
    if (strncmp(line, "cpu ", 4) != 0) {
        fprintf(stderr, "Unexpected format of file %s.\n", PROC_STAT_FILE);
        return -1;
    }

    // Discard the first word, then read the next 8 words containing cpu cycles data
    // Older kernels report fewer columns, the missing ones then stay at zero
    if (sscanf(line, "%*s %llu %llu %llu %llu %llu %llu %llu %llu",
               &cpu_data->cuser, &cpu_data->cnice, &cpu_data->csystem, &cpu_data->cidle,
               &cpu_data->ciowait, &cpu_data->cirq, &cpu_data->csoftirq, &cpu_data->csteal) < 4) {
        fprintf(stderr, "Unexpected format of file %s.\n", PROC_STAT_FILE);
        memset(cpu_data, 0, sizeof(*cpu_data));
        return -1;
    }

    // Calculate cwaiting, cbusy and ctotal
    cpu_data->cwaiting = cpu_data->cidle + cpu_data->ciowait;
    cpu_data->cbusy = cpu_data->cuser + cpu_data->cnice + cpu_data->csystem
                    + cpu_data->cirq + cpu_data->csoftirq + cpu_data->csteal;
    cpu_data->ctotal = cpu_data->cbusy + cpu_data->cwaiting;

    return 0;
}

double cpu_utilization(const struct cpu_cycles_info * before, const struct cpu_cycles_info * after) {
    // Counters are monotonic, but a snapshot that failed to be read is zeroed, so guard the subtraction
    if (after->ctotal <= before->ctotal) {
        return 0.0;
    }

    // Ticks of the interval
    unsigned long long elapsed_time = after->ctotal - before->ctotal;

    // Ticks of the interval the machine spend waiting
    // Some kernel version let it go backward, so it might report more work done than actually done
    unsigned long long waiting_time = 0;
    if (after->cwaiting > before->cwaiting) {
        waiting_time = after->cwaiting - before->cwaiting;
        if (waiting_time > elapsed_time) {
            waiting_time = elapsed_time;
        }
    }

    return (double) (elapsed_time - waiting_time) / (double) elapsed_time;
}
