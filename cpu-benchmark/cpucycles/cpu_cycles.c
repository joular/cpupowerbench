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

#include "cpu_cycles.h"

#define PROC_STAT_FILE "/proc/stat"

void calculate_cpu_cycles(struct cpu_cycles_info * cpu_data) {
    // Open /proc/stat file
    FILE * fp;
    if ((fp = fopen(PROC_STAT_FILE, "r")) == NULL) {
        fprintf(stderr, "Error in reading file %s. Exiting with failure.\n", PROC_STAT_FILE);
        exit(EXIT_FAILURE);
    }

    // Reading cpu cycles from /proc/stat
    // Discard first word, then read the next 4 words containing cpu cycles data
    fscanf(fp, "%*s %lu %lu %lu %lu", &cpu_data->cuser, &cpu_data->cnice, &cpu_data->csystem, &cpu_data->cidle);

    // Close file
    fclose(fp);

    // Calculate cbusy and ctotal
    cpu_data->cbusy = cpu_data->cuser + cpu_data->cnice + cpu_data->csystem;
    cpu_data->ctotal = cpu_data->cuser + cpu_data->cnice + cpu_data->csystem + cpu_data->cidle;
}