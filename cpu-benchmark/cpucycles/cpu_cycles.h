/* Copyright (c) 2020-2023, Université de Pays et des Pays de l'Adour.
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the
# GNU General Public License v3.0 only (GPL-3.0-only)
# which accompanies this distribution, and is available at
# https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Author : Adel Noureddine
*/

#ifndef CPU_CYCLES_H
#define CPU_CYCLES_H

/**
 * Stucture to collect cpu cycles data
 * Data collected from the aggregated "cpu" line of /proc/stat
 * These numbers identify the amount of time the CPU has spent performing different kinds of work. Time units are in USER_HZ or Jiffies (typically hundredths of a second)
 * The two columns that follow those eight, guest and guest_nice, are left out, as the kernel already counts them inside user and nice
 */
struct cpu_cycles_info {
    // user: normal processes executing in user mode
    unsigned long long cuser;
    // nice: niced processes executing in user mode
    unsigned long long cnice;
    // system: processes executing in kernel mode
    unsigned long long csystem;
    // idle: cycles in idle mode
    unsigned long long cidle;
    // iowait: cycles waiting for I/O to complete
    unsigned long long ciowait;
    // irq: cycles servicing hardware interrupts
    unsigned long long cirq;
    // softirq: cycles servicing software interrupts
    unsigned long long csoftirq;
    // steal: cycles stolen by the hypervisor of a virtual machine
    unsigned long long csteal;
    // Waiting cycles : cidle + ciowait
    unsigned long long cwaiting;
    // Busy cycles : cuser + cnice + csystem + cirq + csoftirq + csteal
    unsigned long long cbusy;
    // Total cycles : cbusy + cwaiting (the eight columns)
    unsigned long long ctotal;
};

/**
 * Collect reading from /proc/stat
 * Then calculate CPU cycles : cwaiting, cbusy and ctotal
 * @param cpu_data Snapshot of CPU data read from /proc/stat
 * @return 0 on success, -1 if the statistics could not be read (cpu_data is then zeroed)
 */
int calculate_cpu_cycles(struct cpu_cycles_info * cpu_data);

/**
 * Calculate the CPU utilization ratio between two snapshots
 * @param before Snapshot taken at the start of the interval
 * @param after Snapshot taken at the end of the interval
 * @return CPU utilization of the interval, in the [0.0, 1.0] range
 */
double cpu_utilization(const struct cpu_cycles_info * before, const struct cpu_cycles_info * after);

#endif
