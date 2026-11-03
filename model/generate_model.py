#!/usr/bin/env python3
# Copyright (c) 2026, Adel Noureddine.
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the
# GNU General Public License v3.0 only (GPL-3.0-only)
# which accompanies this distribution, and is available at
# https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Author: Adel Noureddine

"""Fit a power model, watts against CPU usage, from a cpupowerbench run and a power meter log.

Usage: generate_model.py POWER.csv BENCHMARK.csv [--offset S] [--tz ZONE] [--trim S]
                         [--max-degree N] [--time-column N] [--power-column N] [--codename NAME]
"""

import argparse
import json
import re
import sys
import warnings
from math import comb
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

# Seconds the board is taken as idle before and after the run, when lining up the clocks
IDLE_AROUND = 120

# joularcore takes the coefficients of x^0 to x^9
TERMS = 10


def epoch(dates):
    """Seconds since 1970 of time zone aware dates, NaN where there is none."""
    return ((dates - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def parse_times(text, near, tz):
    """Second of each time, counted since 1970 UTC, NaN where a row holds no time.

    Times are seconds since 1970, or milli or microseconds as some meters write, or dates.
    Dates ending with an offset or Z are read as such, the others in the tz zone.
    A date such as 05/10/2026 is read day first if that lands closer to near.
    A reading stamped within a second belongs to the second ending at the next whole one, as the
    benchmark labels its own samples with the end of their second.
    """
    numbers = pd.to_numeric(text, errors="coerce")
    if numbers.notna().mean() > 0.5:
        seconds = numbers.to_numpy(dtype=float)
        while np.nanmedian(seconds) > 1e11:
            seconds = seconds / 1000
        return np.ceil(seconds)
    if text.str.contains(r"(?:Z|[+-]\d\d:?\d\d)$", na=False).mean() > 0.5:
        return np.ceil(epoch(pd.to_datetime(text, errors="coerce", utc=True, format="mixed")))
    best = None
    for dayfirst in (False, True):
        with warnings.catch_warnings():
            # pandas warns when day first does not apply, as with 2026-10-05
            warnings.simplefilter("ignore")
            dates = pd.to_datetime(text, errors="coerce", dayfirst=dayfirst)
            # The hour lived twice when the clocks go back cannot be placed, its rows are dropped
            seconds = np.ceil(epoch(dates.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")))
            if best is None or abs(np.nanmedian(seconds) - near) < abs(np.nanmedian(best) - near):
                best = seconds
    return best


def read_benchmark(path):
    """Board name, bits of its userland and per-second samples of a cpupowerbench CSV."""
    with open(path) as f:
        header = f.readline().lstrip("#").strip()
    # "<board>, <n> cores, governor <g>, kernel <r>, <b> bits", where the board name may hold commas
    tail = re.search(r", \d+ cores, governor \S+, kernel \S+, (\d+) bits$", header)
    if tail is None:
        sys.exit(f"{path}: not a cpupowerbench file")
    # The firmware flags are hexadecimal
    bench = pd.read_csv(path, sep=";", comment="#", dtype={"throttled": str})
    return header[:tail.start()], int(tail.group(1)), bench


def read_power(path, time_column, power_column, near, tz):
    """Mean power of each second of a power meter log."""
    # PowerSpy exports are not UTF-8
    with open(path, encoding="latin-1") as f:
        first = f.readline().rstrip("\r\n")
    sep = next((s for s in "\t;," if s in first), ";")
    fields = first.split(sep)

    # Two columns are time and power (PowerSpyCli and most meters), seven a PowerSpy2 export
    if power_column is None:
        power_column = {2: 2, 7: 4}.get(len(fields))
        if power_column is None:
            sys.exit(f"{path}: {len(fields)} columns separated by '{sep}', "
                     "give the one holding the power with --power-column")
    if not 1 <= time_column <= len(fields) or not 1 <= power_column <= len(fields):
        sys.exit(f"{path} has {len(fields)} columns separated by '{sep}'")
    try:
        float(fields[power_column - 1].replace(",", "."))
        header = None
    except ValueError:
        header = 0

    power = pd.read_csv(path, sep=sep, header=header, encoding="latin-1", dtype=str, skipinitialspace=True)
    # Some exports write 2,945 for 2.945
    watts = pd.to_numeric(power.iloc[:, power_column - 1].str.replace(",", "."), errors="coerce").to_numpy()
    seconds = parse_times(power.iloc[:, time_column - 1].str.strip(), near, tz)
    # Rows without a time or a power (notes, a second header) are left out
    keep = ~np.isnan(seconds) & ~np.isnan(watts)
    if not keep.any():
        sys.exit(f"{path}: no row with both a time and a power")
    # Meters reading faster than once a second are averaged over each second
    return pd.Series(watts[keep]).groupby(seconds[keep].astype(np.int64)).mean()


def correlation(usage, power, lag):
    """Correlation of usage with the power read lag seconds later, and how many seconds both have."""
    shifted = power.reindex(usage.index + lag).to_numpy()
    values = usage.to_numpy()
    common = ~np.isnan(shifted) & ~np.isnan(values)
    if common.sum() < 2:
        return np.nan, 0
    # NaN where one side is constant, as when only the idle taken around the run is in common
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.corrcoef(values[common], shifted[common])[0, 1], common.sum()


def find_offset(usage, power, need):
    """Lag with the best correlation, among all those with at least need seconds in common."""
    best, best_r = None, -1.0
    for lag in range(int(power.index.min() - usage.index.max()), int(power.index.max() - usage.index.min()) + 1):
        r, common = correlation(usage, power, lag)
        if common >= need and r > best_r:
            best, best_r = lag, r
    return best, best_r


def fit(usage, power, degree):
    """Least-squares polynomial that never decreases between 0 and 1, with plain coefficients of x^0 to x^degree.

    Written in Bernstein form, a polynomial whose coefficients never decrease never decreases
    either. Left free, high degrees dip between load steps, and a power that drops as the CPU
    works more makes no sense. Kept rising, they can still follow the jump of the frequency
    governor around 50%, which low degrees smooth over.
    """
    basis = np.stack([comb(degree, k) * usage ** k * (1 - usage) ** (degree - k) for k in range(degree + 1)], axis=1)
    # Column j adds up the Bernstein polynomials from j on, so its weight is the rise of the j-th coefficient
    tails = np.cumsum(basis[:, ::-1], axis=1)[:, ::-1]
    rises = lsq_linear(tails, power, bounds=(np.r_[-np.inf, np.zeros(degree)], np.inf), method="bvls").x
    x = np.polynomial.Polynomial([0, 1])
    model = sum(b * comb(degree, k) * x ** k * (1 - x) ** (degree - k) for k, b in enumerate(np.cumsum(rises)))
    # Polynomial arithmetic drops trailing zero coefficients, as when a slope is clipped to 0
    return np.polynomial.Polynomial(np.r_[model.coef, np.zeros(degree + 1 - len(model.coef))])


def cv_rmse(usage, power, steps, loads, degree):
    """RMSE of every step predicted by a fit on all the other steps, those of the lowest and highest loads left aside."""
    # Seconds of one step are alike, so a degree is judged on whole steps it has not seen.
    # The outermost loads would have to be extrapolated, which no model is meant to do; -l may list them in any order
    errors = []
    for step in pd.Series(loads).groupby(steps).first().sort_values(kind="stable").index[1:-1]:
        out = steps == step
        model = fit(usage[~out], power[~out], degree)
        errors.append(model(usage[out]) - power[out])
    return np.sqrt(np.mean(np.concatenate(errors) ** 2))


def scores(model, usage, power):
    """RMSE in watts and MAPE in percent of a model on the samples."""
    error = model(usage) - power
    return np.sqrt(np.mean(error ** 2)), np.mean(np.abs(error) / power) * 100


def main():
    parser = argparse.ArgumentParser(
        description="Fit a power model, watts against CPU usage, from a cpupowerbench run and a power meter log.")
    parser.add_argument("power", help="power meter log: PowerSpy2 export, or two columns time and power")
    parser.add_argument("benchmark", help="benchmark CSV written by cpupowerbench on the board")
    parser.add_argument("--offset", type=int, metavar="S",
                        help="seconds the power meter clock is ahead of the board, taken as it is instead of "
                             "searching for the best match")
    parser.add_argument("--tz", default="UTC", metavar="ZONE",
                        help="time zone of the dates of the power log when they carry no UTC offset, "
                             "such as Europe/Paris (default UTC)")
    parser.add_argument("--trim", type=int, default=2, metavar="S",
                        help="seconds dropped at the start and end of each step, where loads mix (default 2)")
    parser.add_argument("--max-degree", type=int, default=9, choices=range(1, TERMS), metavar="N",
                        help="highest polynomial degree tried, 1 to 9 (default 9)")
    parser.add_argument("--time-column", type=int, default=1, metavar="N",
                        help="column of the power log holding the time, from 1 (default 1)")
    parser.add_argument("--power-column", type=int, metavar="N",
                        help="column of the power log holding the power in watts, from 1 "
                             "(default 2 for two columns, 4 for a PowerSpy2 export)")
    parser.add_argument("--codename", metavar="NAME",
                        help="name of the model in the JSON file, as in the Power Models Database "
                             "(default: the board name, with -64 on a 64-bit userland)")
    args = parser.parse_args()

    try:
        ZoneInfo(args.tz)
    except (ZoneInfoNotFoundError, ValueError):
        sys.exit(f"Unknown time zone {args.tz}, give a name such as Europe/Paris")

    base = Path(args.benchmark).with_suffix("")
    paths = [f"{base}-data.csv", f"{base}-model.json", f"{base}-model.png"]
    # A previous model is never overwritten, like a previous run by the benchmark
    for path in paths:
        if Path(path).exists():
            sys.exit(f"{path} exists, move it away first")

    board, bits, bench = read_benchmark(args.benchmark)
    power = read_power(args.power, args.time_column, args.power_column, bench.time.median(), args.tz)
    print(f"Board: {board}, {bits} bits")

    # Every load is followed by the same cooldown, so lags a whole step apart match almost as well
    # inside the run. Taking the board as idle around the run sets them apart at its ends.
    b0, b1 = bench.time.min(), bench.time.max()
    idle = bench.usage[bench.load == 0].median()
    usage = pd.concat([pd.Series(idle, index=np.arange(b0 - IDLE_AROUND, b0)),
                       bench.set_index("time").usage,
                       pd.Series(idle, index=np.arange(b1 + 1, b1 + IDLE_AROUND + 1))])
    need = len(bench) / 2
    if args.offset is None:
        offset, r = find_offset(usage, power, need)
    else:
        offset = args.offset
        r, common = correlation(usage, power, offset)
        if common < need:
            sys.exit(f"With --offset {offset}, the power file covers less than half of the benchmark: "
                     "leave it out to search")
    if offset is None:
        sys.exit("The power file covers less than half of the benchmark")
    given = args.offset is not None
    print(f"Clock offset: {offset:+d} s (power meter ahead of the board{', given' if given else ''}), "
          f"correlation {r:.3f}")
    if r < 0.9:
        print("Warning: usage and power hardly match, check the files" + (" and --offset" if given else ""))

    bench["power"] = power.reindex(bench.time + offset).to_numpy()

    # The ondemand governor of Raspberry Pi OS keeps the full frequency 5 s after a busy moment,
    # so the short idle steps after the loads read too high. The idle power comes from the
    # warmup only, once 10 s have passed since whatever ran before the benchmark
    warmup = (bench.step == bench.step.min()) & (bench.load == 0)
    step_time = bench.groupby("step").time
    inside = ((bench.time >= step_time.transform("min") + np.where(warmup, max(10, args.trim), args.trim))
              & (bench.time <= step_time.transform("max") - args.trim))
    data = bench[inside & ((bench.load > 0) | warmup)].dropna(subset=["usage", "power"])
    print(f"Samples: {len(data)} of {len(bench)} seconds, {args.trim} s trimmed at both ends of each load, "
          "idle from the warmup only")
    if not (data.load == 0).any():
        print("Warning: no idle samples, the warmup (-w) is too short: the power at rest is extrapolated")

    # Raspberry Pi firmware flags. Bits 0 to 3, under-voltage, frequency capped, throttled and soft temperature
    # limit, hold at that second. Bits 16 to 19 hold the same since boot, so they also catch shorter events
    flags = bench.throttled.dropna().map(lambda f: int(f, 16))
    if len(flags):
        hit = sorted(bench.load[flags[(flags & 0xF) > 0].index].unique())
        if hit or (flags.iloc[-1] & ~flags.iloc[0]) >> 16:
            print("Warning: the firmware reported under-voltage, frequency capping or throttling "
                  + ("at loads " + ", ".join(f"{load:.0%}" for load in hit) if hit else "during the run")
                  + ", check the power supply and the cooling, then run again")
    hottest = bench.temperature.max()
    if hottest >= 80:
        print(f"Warning: the board reached {hottest:.0f} °C and may have throttled")
    frequency = bench.groupby("step").frequency.median()
    highest = frequency[bench.step[bench.load.idxmax()]]
    if highest < 0.95 * frequency.max():
        print(f"Warning: the CPU ran at {highest:.0f} MHz at the highest load against {frequency.max():.0f} MHz "
              "at a lower one, it may have throttled")

    u, p, steps = data.usage.to_numpy(), data.power.to_numpy(), data.step.to_numpy()
    # Every fit leaves one step out and must keep more load levels than its degree
    max_degree = min(args.max_degree, data.load.nunique() - 2)
    if max_degree < 1:
        sys.exit("Too few load levels to fit a model")

    cv = {}
    print("Degree  CV RMSE (W)")
    for degree in range(1, max_degree + 1):
        cv[degree] = cv_rmse(u, p, steps, data.load.to_numpy(), degree)
        print(f"{degree:6d}  {cv[degree]:11.4f}")
    # The simplest model within 2% of the best one
    degree = min(d for d in cv if cv[d] <= 1.02 * min(cv.values()))

    linear = fit(u, p, 1)
    model = fit(u, p, degree)
    linear_rmse, linear_mape = scores(linear, u, p)
    rmse, mape = scores(model, u, p)
    print(f"Linear: P = {linear.coef[0]:.4f} + {linear.coef[1]:.4f} U, "
          f"RMSE {linear_rmse:.4f} W, CV RMSE {cv[1]:.4f} W, MAPE {linear_mape:.2f} %")
    print(f"Polynomial: degree {degree}, RMSE {rmse:.4f} W, CV RMSE {cv[degree]:.4f} W, MAPE {mape:.2f} %")
    coefficients = model.coef.tolist() + [0.0] * (TERMS - len(model.coef))
    # Ada real literals need a digit on both sides of the point, which 1e-05 lacks
    print("joularcore: (" + ", ".join(np.format_float_positional(c, trim="0") for c in coefficients) + ")")

    data["linear"] = linear(u)
    data["polynomial"] = model(u)
    data.to_csv(paths[0], sep=";", index=False)

    # Laid out like the Power Models Database, the intercept apart from the coefficients of x^1 on
    codename = args.codename or re.sub(r"[^a-z0-9.]", "", board.lower()) + ("-64" if bits == 64 else "")
    result = {codename: {
        "board": board,
        "bits": bits,
        "offset": offset,
        "samples": len(data),
        "linear": {"u": linear.coef[1], "c": linear.coef[0],
                   "rmse": linear_rmse, "cv_rmse": cv[1], "mape": linear_mape},
        "polynomial": {"intercept": model.coef[0], "degree": degree, "coefficients": model.coef[1:].tolist(),
                       "rmse": rmse, "cv_rmse": cv[degree], "mape": mape},
    }}
    with open(paths[1], "w") as f:
        json.dump(result, f, indent=2)

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 9))
    x = np.linspace(0, 1, 201)
    top.scatter(u, p, s=4, alpha=0.3, color="gray", label="samples")
    top.plot(x, linear(x), label="linear")
    top.plot(x, model(x), label=f"degree {degree}")
    top.set(title=board, xlabel="CPU usage", ylabel="Power (W)")
    top.legend()
    elapsed = bench.time - bench.time.iloc[0]
    bottom.plot(elapsed, bench.power, color="tab:red", linewidth=0.8)
    bottom.set_xlabel("Time (s)")
    bottom.set_ylabel("Power (W)", color="tab:red")
    right = bottom.twinx()
    right.plot(elapsed, bench.usage, color="tab:blue", linewidth=0.8)
    right.set_ylabel("CPU usage", color="tab:blue")
    fig.tight_layout()
    fig.savefig(paths[2], dpi=150)
    print("Wrote " + ", ".join(paths))


if __name__ == "__main__":
    main()
