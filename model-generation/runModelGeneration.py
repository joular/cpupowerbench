# Copyright (c) 2020-2026, Adel Noureddine.
# Copyright (c) 2020-2023, Université de Pays et des Pays de l'Adour.
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the
# GNU General Public License v3.0 only (GPL-3.0-only)
# which accompanies this distribution, and is available at
# https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Initial author: Houssam Kanso
# Contributor and maintainer: Adel Noureddine

# Imports for command line arguments
import itertools
import os
import sys
import time
from datetime import timedelta

import matplotlib
# Use a non-interactive backend so the script also runs headless (over SSH, on a server)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# --------------------------------------------
# --------------------------------------------
# --------------------------------------------

# Script variables, change as fit

# CSV file for power data collected from PowerSpy2 meter
POWERSPYCSV = './powerspy.csv'

# CPU cycle data collected from RPiPowerBench benchmark
CPUCYCLESCSV = './cpucycles.csv'

# CPU load data collected from RPiPowerBench benchmark
CPULOADCSV = './cpuload.csv'

# CSV file to save concatenated and cleaned benchmark data
COMPLETEDATACSV = './completeData.csv'

# PNG file to save the plot of the generated power models
MODELPLOTPNG = './powerModel.png'

# Interval in seconds for delays between clocks of PowerSpy2 and Single-Board computers
CLOCKSYNC = 0

# Seconds trimmed at both ends of every load step
# The load generator needs a moment to spawn and its PI controller a moment to settle,
# so the first and last seconds of a step are not yet at the requested load
TRIM_SECONDS = 5

# Date of the experiment, only needed if it cannot be read from the power file
# cpucycles.csv holds a time of day but no date, so the date comes from the power file,
# which is the only one of the three carrying a full date. Set the three variables below
# to override it, for instance for a power file that also holds a time of day only
year = None
month = None
day = None

# --------------------------------------------
# --------------------------------------------
# --------------------------------------------

# Command line arguments

# The layout of the power file is recognised from the file itself, the option is
# still accepted so that older commands keep working
if len(sys.argv) == 2 and sys.argv[1] == "powercsv":
    print("Option powercsv given, the power file layout is detected automatically anyway")

# --------------------------------------------
# --------------------------------------------
# --------------------------------------------

# Utility functions


def detect_delimiter(path, encoding=None):
    """Find the character separating the columns of a CSV file.

    The three files do not agree on one: the benchmark writes cpucycles.csv with
    semicolons and cpuload.csv with commas, and an exported power file uses whichever
    character its software was set to, so read it from the file rather than assume one.
    """
    with open(path, 'r', encoding=encoding or 'utf-8', errors='replace') as f:
        lines = [line for line in itertools.islice(f, 5) if line.strip()]

    if not lines:
        sys.exit(f'Input file is empty: {path}')

    # A separator splits every row in the same number of columns, so it appears the
    # same number of times in each of them, and at least once
    # Tried in this order, so that a decimal comma is not mistaken for a separator
    for candidate in ('\t', ';', ','):
        counts = {line.count(candidate) for line in lines}
        if len(counts) == 1 and counts.pop() > 0:
            return candidate

    sys.exit(f'Cannot tell which character separates the columns of {path}. '
             f'A tab, a semicolon or a comma is expected.')


def read_csv_file(path, encoding=None):
    """Read a CSV file, working out both its separator and whether it has a header row.

    The files written by the benchmark (cpucycles.csv, cpuload.csv) carry no header row,
    while an exported power file usually does. Reading a headerless file with the default
    header=0 silently swallows its first data row, so detect which kind this is.
    """
    if not os.path.isfile(path):
        sys.exit(f'Missing input file: {path}')

    sep = detect_delimiter(path, encoding)

    with open(path, 'r', encoding=encoding or 'utf-8', errors='replace') as f:
        first_line = f.readline()

    # A data row ends with a number (a power reading, a CPU load), a header row ends with a label
    try:
        float(first_line.rstrip('\n').split(sep)[-1])
        header = None
    except (ValueError, IndexError):
        header = 0

    return pd.read_csv(path, sep=sep, encoding=encoding, header=header)


def read_benchmark_csv(path, names, encoding=None):
    """Read one of the CSV files written by the benchmark, and name its columns."""
    data = read_csv_file(path, encoding)
    if len(data.columns) != len(names):
        sys.exit(f'{path} has {len(data.columns)} columns, expected {len(names)}: {names}')
    data.columns = names
    return data


def to_datetimes(timestamps):
    """Turn the timestamp column of a power file into datetimes in the clock of the board.

    A timestamp is either a date and time already, or a number of seconds since the epoch.
    Seconds since the epoch count from UTC, while the board writes its own local time in
    cpucycles.csv, so shift them by the offset this machine's timezone had that day.
    """
    if not pd.api.types.is_numeric_dtype(timestamps):
        return pd.to_datetime(timestamps)

    utc_offset = time.localtime(int(timestamps.iloc[0])).tm_gmtoff
    return pd.to_datetime(timestamps, unit='s') + pd.to_timedelta(utc_offset, unit='s')


def to_experiment_datetime(times, date):
    """Turn a HH:MM:SS column into full datetimes on the date of the experiment.

    The benchmark only records the time of day, so the date is added here. A run that
    crosses midnight makes the clock go backwards, so roll over to the next day when
    it does, rather than folding the whole run onto a single date.
    """
    parsed = pd.to_datetime(times, format='%H:%M:%S')
    # Each backwards step in the time of day is one midnight crossed
    extra_days = (parsed.diff() < timedelta(0)).cumsum()
    return parsed.apply(
        lambda x: x.replace(year=date.year, month=date.month, day=date.day)
    ) + pd.to_timedelta(extra_days, unit='D')


def choose_experiment_date(times, date, power_start, power_end):
    """Pick the date to put on the times of day, the one overlapping the power file most.

    The date comes from the first power reading, but a run started just before midnight
    has its first cpucycles line on the day before, or after it on the day after, so try
    the neighbouring days too and keep whichever lines up with the power readings.
    """
    best_date, best_overlap = date, None
    for shift in (0, -1, 1):
        candidate = date + timedelta(days=shift)
        stamps = to_experiment_datetime(times, candidate)
        overlap = (min(stamps.max(), power_end) - max(stamps.min(), power_start)).total_seconds()
        if best_overlap is None or overlap > best_overlap:
            best_date, best_overlap = candidate, overlap
    return best_date


# --------------------------------------------
# --------------------------------------------
# --------------------------------------------

# First let's process and clean CSV files

print('---------------------------------')
print('Processing and cleaning CSV files')
print('---------------------------------')

# Read the CSV of the wattmeter and the cycles
# The input files are only read, never rewritten, so a run can safely be repeated
wattmeterdata = read_csv_file(POWERSPYCSV, encoding='latin-1')

# Recognise the layout of the power file from the number of columns it holds
if len(wattmeterdata.columns) == 7:
    # A PowerSpy2 file: timestamp, then voltage, current, power, and frequency readings
    print('Power file: PowerSpy2, 7 columns')
    powerspy_file = True
    wattmeterdata.columns = ['Timestamp', 'U RMS', 'I RMS', 'P RMS', 'U Max', 'I Max', 'Frequency']
elif len(wattmeterdata.columns) == 2:
    # A regular CSV with two columns: Timestamp and Power consumption
    print('Power file: two columns, timestamp and power')
    powerspy_file = False
    wattmeterdata.columns = ['Timestamp', 'Power']
else:
    sys.exit(f'{POWERSPYCSV} has {len(wattmeterdata.columns)} columns. Either 7 columns '
             f'(a PowerSpy2 file) or 2 columns (timestamp and power) are expected.')

cyclesdata = read_benchmark_csv(CPUCYCLESCSV, ['TimestampC', 'U'])

# Convert the column from String or seconds to Datetime type
wattmeterdata.Timestamp = to_datetimes(wattmeterdata.Timestamp)

# Synch_time is used to sychronize both files due to the small difference
# found between the clock of the single-board computer and the wattmeter
synch_time = CLOCKSYNC
wattmeterdata.Timestamp = wattmeterdata.Timestamp - timedelta(seconds=synch_time)

# The power file is the only one holding a date, so the date of the experiment comes
# from it, unless it was set by hand at the top of this script
if year and month and day:
    experiment_date = pd.Timestamp(year=year, month=month, day=day).date()
else:
    experiment_date = wattmeterdata.Timestamp.iloc[0].date()

power_start, power_end = wattmeterdata.Timestamp.min(), wattmeterdata.Timestamp.max()
experiment_date = choose_experiment_date(
    cyclesdata.TimestampC, experiment_date, power_start, power_end)
print(f'Date of the experiment: {experiment_date}')

# Create datetime index passing the datetime series
datetime_index = pd.DatetimeIndex(wattmeterdata['Timestamp'].values)
wattmeterdata = wattmeterdata.set_index(datetime_index)

# Drop the column Timestamp not useful anymore
wattmeterdata.drop('Timestamp', axis=1, inplace=True)

# Convert the column from String to Datetime type
cyclesdata.TimestampC = to_experiment_datetime(cyclesdata.TimestampC, experiment_date)

# Create datetime index passing the datetime series
datetime_index = pd.DatetimeIndex(cyclesdata['TimestampC'].values)
cyclesdata = cyclesdata.set_index(datetime_index)

# Clean the data by making sure there no duplicates in the index (timedate)
# that may cause errors while concatenating data
wattmeterdata = wattmeterdata[~wattmeterdata.index.duplicated(keep='first')]
cyclesdata = cyclesdata[~cyclesdata.index.duplicated(keep='first')]

# Concatenate Experimental data and cycle data
result = pd.concat([wattmeterdata, cyclesdata], axis="columns", sort=False, join='inner')

if result.empty:
    # Report the gap between the two files, it is the value CLOCKSYNC has to be set to
    gap = (wattmeterdata.index.min() - cyclesdata.index.min()).total_seconds()
    sys.exit(f'No timestamp is shared by the power file and {CPUCYCLESCSV}.\n'
             f'  power readings : {wattmeterdata.index.min()} to {wattmeterdata.index.max()}\n'
             f'  CPU cycles     : {cyclesdata.index.min()} to {cyclesdata.index.max()}\n'
             f'The two clocks are {gap:.0f} seconds apart, set CLOCKSYNC = {gap:.0f} '
             f'at the top of this script if both files do cover the same run.')

# Drop useless columns and rename columns
if powerspy_file:
    # If using PowerSpy2 data file
    result = result.drop(['U RMS', 'I RMS', 'U Max', 'I Max', 'Frequency'], axis=1)

result.rename(columns={"TimestampC": "Timestamp"}, inplace=True)

# Read the CSV of the times data
timedata = read_benchmark_csv(CPULOADCSV, ['start_time', 'end_time', 'U'])

# Convert the column from String to Datetime type
timedata.start_time = to_experiment_datetime(timedata.start_time, experiment_date)
timedata.end_time = to_experiment_datetime(timedata.end_time, experiment_date)

# Remove the data collected before and after the experiment
powerdata = result
powerdata = powerdata.loc[(powerdata['Timestamp'] >= timedata.iloc[0, 0])]
powerdata = powerdata.loc[(powerdata['Timestamp'] <= timedata.iloc[-1, 1])]

# Remove the waiting time between each stress level,
# and the ramp up and down at both ends of each stress level
steps = []
for i in range(0, timedata.shape[0]):
    newpower = powerdata.loc[
        (powerdata['Timestamp'] > timedata.iloc[i, 0] + timedelta(seconds=TRIM_SECONDS))
        & (powerdata['Timestamp'] < timedata.iloc[i, 1] - timedelta(seconds=TRIM_SECONDS))
    ].copy()
    if newpower.empty:
        print(f'Warning: no data kept for the load step at {timedata.iloc[i, 2]} '
              f'({timedata.iloc[i, 0].time()} - {timedata.iloc[i, 1].time()})')
        continue
    newpower['Real_U'] = timedata.iloc[i, 2]
    steps.append(newpower)

if not steps:
    sys.exit('No usable data left after trimming. Check that cpuload.csv and '
             'cpucycles.csv come from the same run.')

resultdata = pd.concat(steps, ignore_index=True)

# Display the clean data
# If using PowerSpy2 data file
if powerspy_file:
    resultdata.rename(columns={"P RMS": "P"}, inplace=True)
else:
    # If using a regular CSV with two columns: Timestamp and Power consumption
    resultdata.rename(columns={"Power": "P"}, inplace=True)

# Write the results
resultdata.to_csv(COMPLETEDATACSV, index=False, header=True)

print(f'Kept {len(resultdata)} samples over {len(steps)} of the '
      f'{timedata.shape[0]} load steps')
print('Processing and cleaning CSV files finished')
print('File created with complete and clean data: ' + COMPLETEDATACSV)

# --------------------------------------------
# --------------------------------------------
# --------------------------------------------

# Now let's generate power models

print('---------------------------------')
print('Generating power models')
print('---------------------------------')

# Read CSV data file
csvdata = pd.read_csv(COMPLETEDATACSV, sep=',')

# X is the measured CPU utilization, Y the measured power
# Selected by name, so adding a column to completeData.csv cannot silently shift them
X = csvdata[['U']].to_numpy()
Y = csvdata['P'].to_numpy()

#
# Apply Linear Regression
#
print('---------------------------------')
print('Linear Regression')
print('---------------------------------')

# Initialise and fit model
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
r_sq = linear_regressor.score(X, Y)
print('coefficient of determination:', r_sq)

# Print Model Parameters
model_intercept = linear_regressor.intercept_
print('Intercept:', model_intercept)
model_slope = linear_regressor.coef_[0]
print('Slope:', model_slope)
print('Model:', "P = ", model_slope, "* U + ", model_intercept)

# Calculate the estimated power and error % for each point
csvdata['estimated_power'] = linear_regressor.predict(X)
csvdata['error'] = abs(csvdata['estimated_power'] - csvdata['P']) * 100 / csvdata['P']

# Print the average error
print("RMSE = ", root_mean_squared_error(Y, csvdata['estimated_power']))
average_error = csvdata['error'].mean()
print('Linear Regression Average Error:', average_error)

#
# Apply Polynomial Regression
#

print('---------------------------------')
print('Polynomial Regression')
print('---------------------------------')


# Function to find best degree of polynomial regression
def find_best_degree(X, y):
    # random_state keeps the chosen degree reproducible from one run to the next
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    rmses = []
    degrees = np.arange(1, 10)
    min_rmse, min_deg = 1e10, 0

    for deg in degrees:
        # Train features
        poly_features = PolynomialFeatures(degree=deg, include_bias=False)
        x_poly_train = poly_features.fit_transform(x_train)

        # Linear regression
        poly_reg = LinearRegression()
        poly_reg.fit(x_poly_train, y_train)

        # Compare with test data
        # transform, not fit_transform: the test set must use the features fitted on the train set
        x_poly_test = poly_features.transform(x_test)
        poly_predict = poly_reg.predict(x_poly_test)
        poly_rmse = root_mean_squared_error(y_test, poly_predict)
        rmses.append(poly_rmse)

        # Cross-validation of degree
        if min_rmse > poly_rmse:
            min_rmse = poly_rmse
            min_deg = deg

    # Plot and present results
    print('Best degree {} with RMSE {}'.format(min_deg, min_rmse))
    return min_deg


best_degree = find_best_degree(X, Y)

poly_reg = PolynomialFeatures(degree=best_degree, include_bias=False)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, Y)

# Estimate every point in one call, rather than refitting the transform row by row
csvdata['estimated_power_PR'] = pol_reg.predict(X_poly)
csvdata['error_PR'] = abs(csvdata['estimated_power_PR'] - csvdata['P']) * 100 / csvdata['P']

np.set_printoptions(suppress=True)

print('Polynomial coefficients:', pol_reg.coef_)
print('Intercept:', pol_reg.intercept_)
print("RMSE = ", root_mean_squared_error(Y, csvdata['estimated_power_PR']))
print('Polynomial Regression Average Error:', csvdata['error_PR'].mean())

# Save the data with both estimations next to the measured power
csvdata.to_csv(COMPLETEDATACSV, index=False, header=True)

#
# Plot the measured power and both models, to eyeball how well they fit
#

curve_U = np.linspace(csvdata['U'].min(), csvdata['U'].max(), 200).reshape(-1, 1)

plt.figure(figsize=(8, 5))
plt.scatter(csvdata['U'], csvdata['P'], s=4, alpha=0.3, label='Measured')
plt.plot(curve_U, linear_regressor.predict(curve_U),
         label=f'Linear (avg error {average_error:.2f}%)')
plt.plot(curve_U, pol_reg.predict(poly_reg.transform(curve_U)),
         label=f'Polynomial degree {best_degree} '
               f'(avg error {csvdata["error_PR"].mean():.2f}%)')
plt.xlabel('CPU utilization')
plt.ylabel('Power (W)')
plt.title('Power model')
plt.legend()
plt.grid(True)
plt.savefig(MODELPLOTPNG, dpi=100, bbox_inches='tight')
plt.close()

print('Plot of the models saved to: ' + MODELPLOTPNG)
print('Generating power models finished')

# --------------------------------------------
# --------------------------------------------
# --------------------------------------------
