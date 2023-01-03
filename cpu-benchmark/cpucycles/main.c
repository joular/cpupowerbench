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
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "cpu_cycles.h"

int main() {
    struct cpu_cycles_info cci_before, cci_after;
    double utilization = 0.0;

    while (1) {
        calculate_cpu_cycles(&cci_before);
        sleep(1);
        calculate_cpu_cycles(&cci_after);

        utilization = (double) (cci_after.cbusy - cci_before.cbusy) / (double) (cci_after.ctotal - cci_before.ctotal);

        FILE * fp;
        if ((fp = fopen("cpucycles.csv", "a")) == NULL) {
            fprintf(stderr, "Error in reading file. Exiting with failure.\n");
            exit(EXIT_FAILURE);
        }

        time_t t = time(NULL);
        struct tm tm = *localtime(&t);

        fprintf(fp, "%02d:%02d:%02d;%f\n", tm.tm_hour, tm.tm_min, tm.tm_sec, utilization);

        // Close file
        fclose(fp);
    }

    return 0;
}
