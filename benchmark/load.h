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

#ifndef LOAD_H
#define LOAD_H

// Starts one idle thread per CPU this program may use, returns their number or 0 on failure
int load_start(void);

// Share of every CPU to keep busy, from 0 to 1
void load_set(double load);

void load_stop(void);

#endif
