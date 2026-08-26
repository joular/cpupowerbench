/* Copyright (c) 2020-2023, Université de Pays et des Pays de l'Adour.
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the
# GNU General Public License v3.0 only (GPL-3.0-only)
# which accompanies this distribution, and is available at
# https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Author : Adel Noureddine
*/

// Asks for clock_gettime and nanosleep, which are POSIX rather than plain C
#define _POSIX_C_SOURCE 199309L

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cpu_cycles.h"

#define CPUCYCLES_FILE "cpucycles.csv"

// One sample per second, the sampling rate the power model is built on
#define SAMPLING_INTERVAL_SECONDS 1

#define NANOSECONDS_PER_SECOND 1000000000L

/**
 * Sleep until an absolute point of the monotonic clock
 * Sleeping for a fixed duration instead would add the time spent reading /proc/stat and
 * writing the CSV to every interval, and the samples would slowly drift away from the
 * one per second grid they are joined with the power meter readings on
 * @param deadline Point of CLOCK_MONOTONIC to sleep until
 */
static void sleep_until(const struct timespec * deadline) {
    struct timespec now, remaining;

    while (1) {
        clock_gettime(CLOCK_MONOTONIC, &now);

        remaining.tv_sec = deadline->tv_sec - now.tv_sec;
        remaining.tv_nsec = deadline->tv_nsec - now.tv_nsec;
        if (remaining.tv_nsec < 0) {
            remaining.tv_sec -= 1;
            remaining.tv_nsec += NANOSECONDS_PER_SECOND;
        }

        // Deadline already passed, the previous interval overran
        if (remaining.tv_sec < 0) {
            return;
        }

        // Sleeping is interrupted by a signal, sleep out the time that is left
        if (nanosleep(&remaining, NULL) == 0 || errno != EINTR) {
            return;
        }
    }
}

int main() {
    struct cpu_cycles_info cci_before, cci_after;
    double utilization = 0.0;
    struct timespec deadline;

    // First snapshot, it opens the first measured interval
    if (calculate_cpu_cycles(&cci_before) != 0) {
        fprintf(stderr, "Error in reading CPU cycles. Exiting with failure.\n");
        exit(EXIT_FAILURE);
    }

    clock_gettime(CLOCK_MONOTONIC, &deadline);

    while (1) {
        deadline.tv_sec += SAMPLING_INTERVAL_SECONDS;
        sleep_until(&deadline);

        if (calculate_cpu_cycles(&cci_after) != 0) {
            fprintf(stderr, "Error in reading CPU cycles. Exiting with failure.\n");
            exit(EXIT_FAILURE);
        }

        utilization = cpu_utilization(&cci_before, &cci_after);

        FILE * fp;
        if ((fp = fopen(CPUCYCLES_FILE, "a")) == NULL) {
            fprintf(stderr, "Error in writing file %s. Exiting with failure.\n", CPUCYCLES_FILE);
            exit(EXIT_FAILURE);
        }

        time_t t = time(NULL);
        struct tm tm = *localtime(&t);

        fprintf(fp, "%02d:%02d:%02d;%f\n", tm.tm_hour, tm.tm_min, tm.tm_sec, utilization);

        // Close file
        fclose(fp);

        // The end of this interval is the start of the next one, so no cycle is left unmeasured
        cci_before = cci_after;
    }

    return 0;
}
