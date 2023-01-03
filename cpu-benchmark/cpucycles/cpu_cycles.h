/* Copyright (c) 2020-2023, Université de Pays et des Pays de l'Adour.
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the
# GNU General Public License v3.0 only (GPL-3.0-only)
# which accompanies this distribution, and is available at
# https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Author : Adel Noureddine
*/

/**
 * Stucture to collect cpu cycles data
 * Data collected from /proc/stat
 * These numbers identify the amount of time the CPU has spent performing different kinds of work. Time units are in USER_HZ or Jiffies (typically hundredths of a second)
 */
struct cpu_cycles_info {
    // normal processes executing in user mode
    unsigned long cuser;
    // nice: niced processes executing in user mode
    unsigned long cnice;
    // system: processes executing in kernel mode
    unsigned long csystem;
    // idle : cycles in idle mode
    unsigned long cidle;
    // Busy cycles : cuser + cnice + csystem
    unsigned long cbusy;
    // Total cycles : cuser + cnice + csystem + cidle
    unsigned long ctotal;
};

/**
 * Collect reading from /proc/stat
 * Then calculate CPU cycles : cbusy and ctotal
 * @param cpu_data Snapshot of CPU data read from /proc/stat
 */
void calculate_cpu_cycles(struct cpu_cycles_info * cpu_data);
