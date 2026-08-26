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
import os
import sys
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

# Command line arguments

# Variable for power file type
# Options: True for powerspy, False for powercsv
powerspy_file = True

if len(sys.argv) == 2:
    if sys.argv[1] == "powercsv":
        print("Using regular power CSV file")
        powerspy_file = False
    else:
        print("Using PowerSpy2 power file")
else:
    print("Using PowerSpy2 power file")

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

# Change with the date of the experiment
year = 2023
month = 12
day = 21

# --------------------------------------------
# --------------------------------------------
# --------------------------------------------

# Utility functions


def read_benchmark_csv(path, sep, names, encoding=None):
    """Read a benchmark CSV and always give it the column names we expect.

    The files written by the benchmark (cpucycles.csv, cpuload.csv) carry no header
    row, while an exported power meter file may carry one. Reading a headerless file
    with the default header=0 silently swallows its first data row, so detect which
    kind of file this is instead of assuming.
    """
    if not os.path.isfile(path):
        sys.exit(f'Missing input file: {path}')

    with open(path, 'r', encoding=encoding or 'utf-8', errors='replace') as f:
        first_line = f.readline()

    if not first_line.strip():
        sys.exit(f'Input file is empty: {path}')

    # A data row ends with a number (a power reading, a CPU load), a header row ends with a label
    try:
        float(first_line.rstrip('\n').split(sep)[-1])
        header = None
    except (ValueError, IndexError):
        header = 0

    data = pd.read_csv(path, sep=sep, encoding=encoding, header=header)
    if len(data.columns) != len(names):
        sys.exit(f'{path} has {len(data.columns)} columns, expected {len(names)}: {names}')
    data.columns = names
    return data


def to_experiment_datetime(times):
    """Turn a HH:MM:SS column into full datetimes on the date of the experiment.

    The benchmark only records the time of day, so the date is added here. A run that
    crosses midnight makes the clock go backwards, so roll over to the next day when
    it does, rather than folding the whole run onto a single date.
    """
    parsed = pd.to_datetime(times, format='%H:%M:%S')
    # Each backwards step in the time of day is one midnight crossed
    extra_days = (parsed.diff() < timedelta(0)).cumsum()
    return parsed.apply(
        lambda x: x.replace(year=year, month=month, day=day)
    ) + pd.to_timedelta(extra_days, unit='D')


# --------------------------------------------
# --------------------------------------------
# --------------------------------------------

# First let's process and clean CSV files

print('---------------------------------')
print('Processing and cleaning CSV files')
print('---------------------------------')

# Read the CSV of the wattmeter and the cycles
# The input files are only read, never rewritten, so a run can safely be repeated
if powerspy_file:
    # If using PowerSpy2 data file
    wattmeterdata = read_benchmark_csv(
        POWERSPYCSV, '\t',
        ['Timestamp', 'U RMS', 'I RMS', 'P RMS', 'U Max', 'I Max', 'Frequency'],
        encoding='latin-1')
else:
    # If using a regular CSV with two columns: Timestamp and Power consumption
    wattmeterdata = read_benchmark_csv(
        POWERSPYCSV, ',', ['Timestamp', 'Power'], encoding='latin-1')

cyclesdata = read_benchmark_csv(CPUCYCLESCSV, ';', ['TimestampC', 'U'])

# Convert the column from String to Datetime type
if powerspy_file:
    # If using PowerSpy2 data file
    wattmeterdata.Timestamp = pd.to_datetime(wattmeterdata.Timestamp)
else:
    # If using a regular CSV with two columns: Timestamp and Power consumption
    # Timestamp is usually in seconds
    wattmeterdata.Timestamp = pd.to_datetime(wattmeterdata.Timestamp, unit='s')
    # If time is shifted by 2 hours, then fix it
    # wattmeterdata.Timestamp = wattmeterdata.Timestamp + pd.to_timedelta(2, unit='h')

# Synch_time is used to sychronize both files due to the small difference
# found between the clock of the single-board computer and the wattmeter
synch_time = CLOCKSYNC
wattmeterdata.Timestamp = wattmeterdata.Timestamp - timedelta(seconds=synch_time)

# Create datetime index passing the datetime series
datetime_index = pd.DatetimeIndex(wattmeterdata['Timestamp'].values)
wattmeterdata = wattmeterdata.set_index(datetime_index)

# Drop the column Timestamp not useful anymore
wattmeterdata.drop('Timestamp', axis=1, inplace=True)

# Convert the column from String to Datetime type
cyclesdata.TimestampC = to_experiment_datetime(cyclesdata.TimestampC)

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
    sys.exit('No timestamp is shared by the power file and cpucycles.csv. '
             'Check that both cover the same run, and adjust CLOCKSYNC if the '
             'power meter clock is offset from the board clock.')

# Drop useless columns and rename columns
if powerspy_file:
    # If using PowerSpy2 data file
    result = result.drop(['U RMS', 'I RMS', 'U Max', 'I Max', 'Frequency'], axis=1)

result.rename(columns={"TimestampC": "Timestamp"}, inplace=True)

# Read the CSV of the times data
timedata = read_benchmark_csv(CPULOADCSV, ',', ['start_time', 'end_time', 'U'])

# Convert the column from String to Datetime type
timedata.start_time = to_experiment_datetime(timedata.start_time)
timedata.end_time = to_experiment_datetime(timedata.end_time)

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
