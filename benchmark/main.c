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

#include <errno.h>
#include <glob.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/timex.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#include "load.h"
#include "usage.h"

#define MAX_LOADS 100
#define MAX_CPUS 1024

#define TEMPERATURE_FILE "/sys/class/thermal/thermal_zone0/temp"
#define FREQUENCY_FILE "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
// Raspberry Pi firmware flags for under-voltage, frequency capping and throttling, now and since boot
#define THROTTLED_FILE "/sys/devices/platform/soc*/soc*:firmware/get_throttled"

struct step {
    double load;
    int seconds;
};

// An idle warmup, then every load followed by an idle cooldown
static struct step steps[2 * MAX_LOADS + 1];
static int step_count;

static volatile sig_atomic_t stop;

static void on_signal(int number) {
    (void) number;
    stop = 1;
}

static void fail(const char *message) {
    fprintf(stderr, "cpupowerbench: %s\n", message);
    exit(EXIT_FAILURE);
}

static void print_usage(FILE *stream) {
    fprintf(stream,
            "Usage: cpupowerbench [options]\n"
            "  -w SECONDS  idle time before the first load (default 30)\n"
            "  -d SECONDS  time at each load (default 30)\n"
            "  -c SECONDS  idle time after each load (default 5)\n"
            "  -s PERCENT  loads go from this percentage to 100%% in steps of it (default 5)\n"
            "  -l LOADS    loads to run instead of -s, from 0.001 to 1, comma separated (e.g. 0.5,1)\n"
            "  -o FILE     output file (default benchmark-<host>-<date>.csv)\n");
}

static int parse_int(const char *text, int min, int max, const char *option) {
    char *end;
    long value = strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < min || value > max) {
        fprintf(stderr, "cpupowerbench: %s takes a whole number from %d to %d\n", option, min, max);
        exit(EXIT_FAILURE);
    }
    return (int) value;
}

static int parse_loads(const char *text, double *loads) {
    int count = 0;
    char *end;
    do {
        if (count == MAX_LOADS) {
            fail("-l takes at most 100 loads");
        }
        loads[count] = strtod(text, &end);
        // Below 0.001 the load would round to nothing
        if (end == text || (*end != ',' && *end != '\0') || loads[count] < 0.001 || loads[count] > 1.0) {
            fail("-l takes loads from 0.001 to 1, separated by commas");
        }
        count++;
        text = end + 1;
    } while (*end == ',');
    return count;
}

static void add_step(double load, int seconds) {
    steps[step_count].load = load;
    steps[step_count].seconds = seconds;
    step_count++;
}

// First line of a small system file without its newline, empty if it cannot be read
static void read_text(const char *path, char *text, size_t size) {
    FILE *file = fopen(path, "r");
    text[0] = '\0';
    if (file != NULL) {
        if (fgets(text, (int) size, file) == NULL) {
            text[0] = '\0';
        }
        fclose(file);
    }
    text[strcspn(text, "\n")] = '\0';
}

// NAN when the board does not report it
static double read_number(const char *path, double divisor) {
    char text[32], *end;
    read_text(path, text, sizeof(text));
    double value = strtod(text, &end);
    return end == text ? NAN : value / divisor;
}

// Returns 0 once the benchmark is stopped
static int sleep_until(const struct timespec *time) {
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, time, NULL) == EINTR) {
        if (stop) {
            return 0;
        }
    }
    return !stop;
}

static long long wall_clock_second(void) {
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    // Samples are taken just after a whole second, rounding keeps their label steady
    return (long long) now.tv_sec + (now.tv_nsec >= 500000000L);
}

int main(int argc, char *argv[]) {
    int warmup = 30, duration = 30, cooldown = 5, increment = 0;
    double loads[MAX_LOADS];
    int load_count = 0;
    const char *output = NULL;
    int option;

    while ((option = getopt(argc, argv, "w:d:c:s:l:o:h")) != -1) {
        switch (option) {
        case 'w': warmup = parse_int(optarg, 0, 3600, "-w"); break;
        case 'd': duration = parse_int(optarg, 1, 3600, "-d"); break;
        case 'c': cooldown = parse_int(optarg, 0, 3600, "-c"); break;
        case 's': increment = parse_int(optarg, 1, 100, "-s"); break;
        case 'l': load_count = parse_loads(optarg, loads); break;
        case 'o': output = optarg; break;
        case 'h': print_usage(stdout); return EXIT_SUCCESS;
        default: print_usage(stderr); return EXIT_FAILURE;
        }
    }
    if (optind < argc) {
        print_usage(stderr);
        return EXIT_FAILURE;
    }

    if (load_count > 0 && increment > 0) {
        fail("-l and -s cannot be given together");
    }
    if (load_count == 0) {
        increment = increment > 0 ? increment : 5;
        for (int percent = increment; percent < 100; percent += increment) {
            loads[load_count++] = percent / 100.0;
        }
        loads[load_count++] = 1.0;
    }

    // The idle steps give the model its power at rest, and clear edges to line up the power meter with
    if (warmup > 0) {
        add_step(0.0, warmup);
    }
    for (int i = 0; i < load_count; i++) {
        add_step(loads[i], duration);
        if (cooldown > 0) {
            add_step(0.0, cooldown);
        }
    }

    int left = 0;
    for (int s = 0; s < step_count; s++) {
        left += steps[s].seconds;
    }

    // CPU times at the last sample, and at the start of the step for each CPU
    struct cpu_times before, after;
    static struct cpu_times cpus[MAX_CPUS], step_cpus[MAX_CPUS];
    int cpu_count = read_cpu_times(&before, cpus, MAX_CPUS);
    if (cpu_count < 0) {
        fail("cannot read /proc/stat, the benchmark runs on Linux only");
    }

    char name[512];
    if (output == NULL) {
        char host[256];
        char date[32];
        time_t now = time(NULL);
        if (gethostname(host, sizeof(host)) != 0) {
            strcpy(host, "board");
        }
        // The short name, without the domain
        host[sizeof(host) - 1] = '\0';
        host[strcspn(host, ".")] = '\0';
        strftime(date, sizeof(date), "%Y%m%d-%H%M%S", localtime(&now));
        snprintf(name, sizeof(name), "benchmark-%s-%s.csv", host, date);
        output = name;
    }

    // Never overwrite a previous run
    FILE *csv = fopen(output, "wx");
    if (csv == NULL) {
        fprintf(stderr, "cpupowerbench: cannot create %s: %s\n", output, strerror(errno));
        return EXIT_FAILURE;
    }

    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = on_signal;
    sigaction(SIGINT, &action, NULL);
    sigaction(SIGTERM, &action, NULL);

    // The load threads block the signals, so they always wake the main thread from its sleep
    sigset_t signals, previous;
    sigemptyset(&signals);
    sigaddset(&signals, SIGINT);
    sigaddset(&signals, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &signals, &previous);
    int cores = load_start();
    pthread_sigmask(SIG_SETMASK, &previous, NULL);
    if (cores == 0) {
        fclose(csv);
        unlink(output);
        fail("cannot start the load threads");
    }

    char model[128], governor[64];
    struct utsname kernel;
    uname(&kernel);
    read_text("/proc/device-tree/model", model, sizeof(model));
    if (model[0] == '\0') {
        read_text("/sys/devices/virtual/dmi/id/product_name", model, sizeof(model));
    }
    if (model[0] == '\0') {
        snprintf(model, sizeof(model), "%s", kernel.machine);
    }
    read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", governor, sizeof(governor));
    if (governor[0] == '\0') {
        strcpy(governor, "unknown");
    }

    // The flags come in hexadecimal, as the firmware writes them
    char throttled[32] = "";
    glob_t found;
    if (glob(THROTTLED_FILE, 0, NULL, &found) == 0) {
        read_text(found.gl_pathv[0], throttled, sizeof(throttled));
        globfree(&found);
    }

    // The bits are those of the userland, which has its own models on some boards
    char info[512];
    snprintf(info, sizeof(info), "%s, %d cores, governor %s, kernel %s, %d bits%s%s", model, cores, governor,
             kernel.release, (int) (sizeof(void *) * 8), throttled[0] != '\0' ? ", throttled " : "", throttled);

    fprintf(csv, "# %s\n", info);
    fprintf(csv, "time;usage;load;step;temperature;frequency\n");
    fflush(csv);

    printf("Board: %s\n", info);

    // Usage is measured over the whole board, so it could not reach 100%
    long online = sysconf(_SC_NPROCESSORS_ONLN);
    if (cores < online) {
        printf("Warning: only %d of the %ld CPUs can be loaded (affinity, cpuset), usage will stay below %.0f%%\n",
               cores, online, 100.0 * cores / (double) online);
    }

    // Read only, the power meter and this board are joined on their clocks
    struct timex clock_state;
    memset(&clock_state, 0, sizeof(clock_state));
    if (ntp_adjtime(&clock_state) == TIME_ERROR) {
        printf("Warning: the clock is not synchronized, its time may not match the power meter's\n");
    }

    printf("Writing %s\n", output);
    printf("%d steps, %d min %02d s. Ctrl+C stops and keeps what was measured.\n",
           step_count, left / 60, left % 60);
    fflush(stdout);

    // Tick on the whole seconds of the wall clock, the ones the power meter stamps its readings with.
    // The monotonic clock keeps the ticks one second apart even if the wall clock is set meanwhile
    struct timespec tick, now;
    clock_gettime(CLOCK_REALTIME, &now);
    clock_gettime(CLOCK_MONOTONIC, &tick);
    tick.tv_nsec += 1000000000L - now.tv_nsec;
    if (tick.tv_nsec >= 1000000000L) {
        tick.tv_nsec -= 1000000000L;
        tick.tv_sec++;
    }
    if (sleep_until(&tick) && (cpu_count = read_cpu_times(&before, cpus, MAX_CPUS)) < 0) {
        fail("cannot read /proc/stat");
    }

    for (int s = 0; s < step_count && !stop; s++) {
        double usage_sum = 0.0, temperature_sum = 0.0, frequency_sum = 0.0;
        int samples = 0;
        int step_cpu_count = cpu_count;
        memcpy(step_cpus, cpus, sizeof(cpus[0]) * (size_t) cpu_count);

        // Set right after a sample, so every sample of a step is taken at its load
        load_set(steps[s].load);

        for (int i = 0; i < steps[s].seconds; i++) {
            tick.tv_sec++;
            if (!sleep_until(&tick)) {
                break;
            }
            cpu_count = read_cpu_times(&after, cpus, MAX_CPUS);
            if (cpu_count < 0) {
                fprintf(stderr, "cpupowerbench: cannot read /proc/stat\n");
                stop = 1;
                break;
            }

            double usage = cpu_usage(&before, &after);
            double temperature = read_number(TEMPERATURE_FILE, 1000.0);
            double frequency = read_number(FREQUENCY_FILE, 1000.0);
            before = after;

            fprintf(csv, "%lld;%.4f;%.3f;%d;", wall_clock_second(), usage, steps[s].load, s);
            if (!isnan(temperature)) {
                fprintf(csv, "%.1f", temperature);
            }
            fputc(';', csv);
            if (!isnan(frequency)) {
                fprintf(csv, "%.0f", frequency);
            }
            fputc('\n', csv);
            // Written as it goes, so an interrupted run keeps its data
            fflush(csv);

            usage_sum += usage;
            temperature_sum += temperature;
            frequency_sum += frequency;
            samples++;
            left--;
        }
        if (samples == 0) {
            break;
        }

        printf("Step %d/%d: ", s + 1, step_count);
        if (steps[s].load > 0.0) {
            printf("load %.4g%%", steps[s].load * 100.0);
        } else {
            printf("idle");
        }
        printf(", measured %.1f%%", usage_sum / samples * 100.0);

        // Every CPU should carry the same load, not a few of them all of it
        double lowest = 1.0, highest = 0.0;
        for (int c = 0; c < step_cpu_count && c < cpu_count; c++) {
            double usage = cpu_usage(&step_cpus[c], &cpus[c]);
            lowest = usage < lowest ? usage : lowest;
            highest = usage > highest ? usage : highest;
        }
        if (step_cpu_count > 1 && cpu_count > 1) {
            printf(" (each CPU %.1f to %.1f%%)", lowest * 100.0, highest * 100.0);
        }

        if (!isnan(temperature_sum)) {
            printf(", %.1f °C", temperature_sum / samples);
        }
        if (!isnan(frequency_sum)) {
            printf(", %.0f MHz", frequency_sum / samples);
        }
        printf(", %d:%02d left\n", left / 60, left % 60);

        // Other programs would add their own power to every step of the model
        if (s == 0 && warmup > 0 && usage_sum / samples > 0.05) {
            printf("Warning: the CPU was %.1f%% busy while idle, stop other programs for an accurate model\n",
                   usage_sum / samples * 100.0);
        }
        fflush(stdout);
    }

    load_stop();
    fclose(csv);

    if (stop) {
        printf("Benchmark interrupted, what was measured is in %s\n", output);
        return EXIT_FAILURE;
    }
    printf("Benchmark finished, results in %s\n", output);
    return EXIT_SUCCESS;
}
