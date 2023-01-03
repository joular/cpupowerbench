#[[
# Copyright (c) 2020-2023, Université de Pays et des Pays de l'Adour.
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the
# GNU General Public License v3.0 only (GPL-3.0-only)
# which accompanies this distribution, and is available at
# https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Author : Adel Noureddine
]]

cmake_minimum_required(VERSION 3.7)
project(cpucycles C)

set(CMAKE_C_STANDARD 11)

add_executable(cpucycles main.c cpu_cycles.h cpu_cycles.c)