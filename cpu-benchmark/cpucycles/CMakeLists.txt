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

cmake_minimum_required(VERSION 3.10)
project(cpucycles C)

set(CMAKE_C_STANDARD 11)

# Build optimised unless another build type is asked for on the command line
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Put the program next to this file, so it stays at cpucycles/cpucycles
# whether it is built in place or in a separate build folder
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(cpucycles main.c cpu_cycles.h cpu_cycles.c)

target_compile_options(cpucycles PRIVATE -Wall -Wextra)
