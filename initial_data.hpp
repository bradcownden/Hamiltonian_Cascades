// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Anxo Biasi, Brad Cownden, Oleg Evnin, Alvaro Iturbe Jabaloyes
//
// Declaration of make_initial_data().
// See README.md and LICENSE for details.

#include "complex128.hpp"
#include "main.cuh"

void make_initial_data(const ID_parameters& id_params, const char* A_load_path, const char* B_load_path, const char* f_load_path, const char* logf_load_path, double* hostA, double* hostB, double* host_f, double* host_logf, const int ID_type);