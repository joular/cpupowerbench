# Copyright (c) 2020-2023, Université de Pays et des Pays de l'Adour.
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the
# GNU General Public License v3.0 only (GPL-3.0-only)
# which accompanies this distribution, and is available at
# https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Author: Houssam Kanso
#
# Contributors: Adel Noureddine

# Imports for processing and cleaning csv files
import pandas as pd
from datetime import timedelta

# Imports to generate power models
import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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

# Interval in seconds for delays between clocks of PowerSpy2 and Single-Board computers
CLOCKSYNC = 0

# --------------------------------------------
# --------------------------------------------
# --------------------------------------------

## First let's process and clean CSV files

print('---------------------------------')
print('Processing and cleaning CSV files')
print('---------------------------------')

# Read CSV files and rewrite them with proper headers

## If using PowerSpy2 data file
temp_df = pd.read_csv(POWERSPYCSV,sep='\t',encoding = 'latin-1')
temp_df.columns = ['Timestamp','U RMS','I RMS','P RMS','U Max','I Max','Frequency']
temp_df.to_csv(POWERSPYCSV,sep='\t', index = False, header=True)

## If using a regular CSV with two columns: Timestamp and Power consumption
# temp_df = pd.read_csv(POWERSPYCSV,sep=',',encoding = 'latin-1')
# temp_df.columns = ['Timestamp','Power']
# temp_df.to_csv(POWERSPYCSV,sep=',', index = False, header=True)

temp_df = pd.read_csv(CPUCYCLESCSV,sep=';')
temp_df.columns = ['TimestampC','U']#,'P_unused']
temp_df.to_csv(CPUCYCLESCSV,sep=';', index = False, header=True )

temp_df = pd.read_csv(CPULOADCSV,sep=',') 
temp_df.columns = ['start_time','end_time','U']
temp_df.to_csv(CPULOADCSV,sep=',', index = False, header=True)

# Read the CSV of the wattmeter and the cycles
## If using PowerSpy2 data file
wattmeterdata = pd.read_csv(POWERSPYCSV,sep='\t',encoding = 'latin-1')
wattmeterdata.columns = ['Timestamp','U RMS','I RMS','P RMS','U Max','I Max','Frequency']

## If using a regular CSV with two columns: Timestamp and Power consumption
# wattmeterdata = pd.read_csv(POWERSPYCSV,sep=',',encoding = 'latin-1')
#wattmeterdata.columns = ['Timestamp','Power']

cyclesdata = pd.read_csv(CPUCYCLESCSV,sep=';')
cyclesdata.columns = ['TimestampC','U']

# Convert the column from String to Datetime type
## If using PowerSpy2 data file
wattmeterdata.Timestamp=pd.to_datetime(wattmeterdata.Timestamp)
## If using a regular CSV with two columns: Timestamp and Power consumption
## Timestamp is usually in seconds
# wattmeterdata.Timestamp=pd.to_datetime(wattmeterdata.Timestamp, unit='s')

#Synch_time is used to sychronize both files due to the small difference found between the clock of the single-board computer and the wattmeter
synch_time=CLOCKSYNC
wattmeterdata.Timestamp=wattmeterdata.Timestamp-timedelta(seconds=synch_time)

# Create datetime index passing the datetime series
datetime_series = pd.to_datetime(wattmeterdata['Timestamp'])
datetime_index = pd.DatetimeIndex(datetime_series.values)
wattmeterdata = wattmeterdata.set_index(datetime_index)

# Drop the column Timestamp not useful anymore
wattmeterdata.drop('Timestamp',axis=1,inplace=True)

# Convert the column from String to Datetime type
cyclesdata.TimestampC=pd.to_datetime(cyclesdata.TimestampC)

# Create datetime index passing the datetime series
datetime_series = pd.to_datetime(cyclesdata['TimestampC'])
datetime_index = pd.DatetimeIndex(datetime_series.values)
cyclesdata = cyclesdata.set_index(datetime_index)

# Clean the data by making sure there no duplicates in the index (timedate) that may cause errors while concatenating data
wattmeterdata = wattmeterdata[~wattmeterdata.index.duplicated(keep='first')]
wattmeterdata.drop_duplicates()
cyclesdata = cyclesdata[~cyclesdata.index.duplicated(keep='first')]
cyclesdata.drop_duplicates()

# Concatenate Experimental data and cycle data
result = pd.concat([wattmeterdata, cyclesdata], axis=1, sort=False, join='inner')

# Drop useless columns and rename columns
## If using PowerSpy2 data file
result=result.drop(['U RMS', 'I RMS' , 'U Max', 'I Max','Frequency'], axis=1)

result.rename(columns={"TimestampC": "Timestamp"}, inplace=True)

# Read the CSV of the times data
timedata = pd.read_csv(CPULOADCSV,sep=',')
timedata.columns = ['start_time','end_time','U']

# Remove the data collected before and after the experiment
powerdata = result
powerdata = powerdata.loc[(powerdata['Timestamp'] >= timedata.iloc[0,0])]
powerdata = powerdata.loc[(powerdata['Timestamp'] <= timedata.iloc[-1,1])]

# Convert the column from String to Datetime type
timedata.start_time=pd.to_datetime(timedata.start_time)
timedata.end_time=pd.to_datetime(timedata.end_time)

# Remove the waiting time between each stress level
resultdata=pd.DataFrame()
for i in range(0, timedata.shape[0]):
    newpower = powerdata.loc[(powerdata['Timestamp'] > timedata.iloc[i,0]+timedelta(seconds=3))]
    newpower = newpower.loc[(powerdata['Timestamp'] < timedata.iloc[i,1]-timedelta(seconds=3))]
    newpower['Real_U']=timedata.iloc[i,2]
    resultdata=resultdata.append(newpower, ignore_index=True)
cleandata=resultdata

# Display the clean data
## If using PowerSpy2 data file
resultdata.rename(columns={"P RMS": "P"}, inplace=True)
## If using a regular CSV with two columns: Timestamp and Power consumption
resultdata.rename(columns={"Power": "P"}, inplace=True)

# Write the results
resultdata.to_csv (r""+COMPLETEDATACSV, index = False, header=True)

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
X = csvdata['U']
Y = csvdata['P']

#
# Apply Linear Regression
#
print('---------------------------------')
print('Linear Regression')
print('---------------------------------')

# Initialise and fit model
X = csvdata.iloc[:, 2].values.reshape(-1, 1)  # values converts it into a numpy array
Y = csvdata.iloc[:, 0].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column

linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
r_sq = linear_regressor.score(X, Y)
print('coefficient of determination:', r_sq)

# Print Model Parameters
model_intercept = linear_regressor.intercept_[0]
print('Intercept:', model_intercept)
model_slope = linear_regressor.coef_[0][0]
print('Slope:', model_slope)
print('Model:', "P = ", linear_regressor.coef_[0][0] , "* U + ", linear_regressor.intercept_[0])

# Calculate the estimated power and error % for each point
csvdata['estimated_power'] = linear_regressor.coef_[0][0] * csvdata['U'] + linear_regressor.intercept_[0]
csvdata['error'] = abs (csvdata['estimated_power'] - csvdata['P']) * 100 / csvdata['P']

# Print the average error
print("RMSE = ", math.sqrt(mean_squared_error(Y, linear_regressor.coef_[0][0] * csvdata['U'] + linear_regressor.intercept_[0])))
average_error = csvdata.mean().error
print('Linear Regression Average Error:', average_error)

#
# Apply Polynomial Regression
#

print('---------------------------------')
print('Polynomial Regression')
print('---------------------------------')

# Function to find best degree of polynomial regression
def find_best_degree(X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
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
        x_poly_test = poly_features.fit_transform(x_test)
        poly_predict = poly_reg.predict(x_poly_test)
        poly_mse = mean_squared_error(y_test, poly_predict)
        poly_rmse = np.sqrt(poly_mse)
        rmses.append(poly_rmse)

        # Cross-validation of degree
        if min_rmse > poly_rmse:
            min_rmse = poly_rmse
            min_deg = deg

    # Plot and present results
    print('Best degree {} with RMSE {}'.format(min_deg, min_rmse))
    return min_deg

best_degree = find_best_degree(X,Y)

poly_reg = PolynomialFeatures(degree=best_degree)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, Y)

csvdata["estimated_power_PR"] = ""
csvdata["error_PR"] = ""

for i in range(0, csvdata.shape[0]):
    csvdata.loc[i,'estimated_power_PR'] = pol_reg.predict(poly_reg.fit_transform([[csvdata.loc[i,'U']]]))[0][0]
    csvdata.loc[i,'error_PR'] = abs (csvdata.loc[i,'estimated_power_PR'] - csvdata.loc[i,'P']) * 100 / csvdata.loc[i,'P']

np.set_printoptions(suppress=True)

print('Polynomial coefficients:', pol_reg.coef_[0])
print('Intercept:', pol_reg.intercept_[0])
print('Polynomial Regression Average Error:', csvdata.mean().error_PR)

print('Generating power models finished')

# --------------------------------------------
# --------------------------------------------
# --------------------------------------------
