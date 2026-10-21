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

#define _GNU_SOURCE

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#include "load.h"

/* Each period works first and sleeps for the rest.
Periods of 10 ms on average look like a steady load to the frequency governor, rather than bursts at full speed. Their length is drawn between 5 and 15 ms: with a fixed length, the work would meet the kernel's accounting tick at the same point every time, and /proc/stat would misreport the usage
*/

#define MIN_PERIOD_NS 5000000L
#define PERIOD_SPREAD_NS 10000000U

// How often an idle thread checks for a new load, rarely enough to let the CPU rest
#define IDLE_NS 100000000L

static atomic_int load_permille;
static atomic_int running;
static pthread_t *threads;
static int thread_count;

static void add_ns(struct timespec *time, long ns) {
    time->tv_nsec += ns;
    while (time->tv_nsec >= 1000000000L) {
        time->tv_nsec -= 1000000000L;
        time->tv_sec++;
    }
}

static int earlier(const struct timespec *a, const struct timespec *b) {
    return a->tv_sec < b->tv_sec || (a->tv_sec == b->tv_sec && a->tv_nsec < b->tv_nsec);
}

// xorshift, plenty for spreading period lengths
static unsigned int next_random(unsigned int *state) {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

static void *load_thread(void *seed) {
    unsigned int state = (unsigned int) (uintptr_t) seed;
    struct timespec period_end, busy_until, now;

    clock_gettime(CLOCK_MONOTONIC, &period_end);
    while (atomic_load(&running)) {
        int permille = atomic_load(&load_permille);
        if (permille == 0) {
            add_ns(&period_end, IDLE_NS);
            clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &period_end, NULL);
            continue;
        }

        long period = MIN_PERIOD_NS + (long) (next_random(&state) % PERIOD_SPREAD_NS);

        // Timed from the actual wake up, so a late one does not cut the work short
        clock_gettime(CLOCK_MONOTONIC, &busy_until);
        add_ns(&busy_until, period / 1000 * permille);
        do {
            clock_gettime(CLOCK_MONOTONIC, &now);
        } while (earlier(&now, &busy_until));

        // Sleeping until an absolute time keeps the share of work exact over many periods
        add_ns(&period_end, period);
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &period_end, NULL);
    }
    return NULL;
}

int load_start(void) {
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return 0;
    }

    int count = CPU_COUNT(&allowed);
    threads = calloc(count, sizeof(pthread_t));
    if (threads == NULL) {
        return 0;
    }

    atomic_store(&running, 1);
    for (int cpu = 0; cpu < CPU_SETSIZE && thread_count < count; cpu++) {
        if (!CPU_ISSET(cpu, &allowed)) {
            continue;
        }
        // Any odd multiple gives every thread its own non-zero seed
        uintptr_t seed = (uintptr_t) (cpu + 1) * 2654435761U;
        if (pthread_create(&threads[thread_count], NULL, load_thread, (void *) seed) != 0) {
            load_stop();
            return 0;
        }

        // One thread per CPU, so the scheduler cannot stack two of them on the same one.
        // Where pinning is refused the threads still run, only less evenly spread
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(cpu, &one);
        pthread_setaffinity_np(threads[thread_count], sizeof(one), &one);
        thread_count++;
    }
    return thread_count;
}

void load_set(double load) {
    atomic_store(&load_permille, (int) (load * 1000.0 + 0.5));
}

void load_stop(void) {
    atomic_store(&running, 0);
    for (int i = 0; i < thread_count; i++) {
        pthread_join(threads[i], NULL);
    }
    free(threads);
    threads = NULL;
    thread_count = 0;
}
