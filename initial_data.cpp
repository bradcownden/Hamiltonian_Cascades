// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Anxo Biasi, Brad Cownden, Oleg Evnin, Alvaro Iturbe Jabaloyes
//
// Generates the initial data (f[n] profile and A, B mode amplitudes) in
// high precision and writes it to binary files.
// See README.md and LICENSE for details.

#include "initial_data.hpp"
#include <cmath>

// Test for NaN or Inf values in f array
static inline bool is_bad_double(double x)
{
  return std::isnan(x) || !std::isfinite(x);
}

// Debug scan for NaN or Inf values in an array of doubles
static void debug_scan_array(const char* name, const double* data, int count)
{
  int bad_count = 0;
  for (int i = 0; i < count; ++i) {
    if (is_bad_double(data[i])) {
      fprintf(stderr, "[NaN-DEBUG] %s[%d] = %.17g\n", name, i, data[i]);
      bad_count++;
      if (bad_count >= 20) {
        fprintf(stderr, "[NaN-DEBUG] %s: stopping report after 20 invalid values\n", name);
        break;
      }
    }
  }

  if (bad_count == 0) {
    fprintf(stderr, "[NaN-DEBUG] %s: no NaN/Inf values found\n", name);
  }
}

// Create initial data at high precision and copy double values to host
// pointers. Write initial data to binary files for later loading and debugging.

void make_initial_data(const ID_parameters& id_params, const char* A_load_path, const char* B_load_path, const char* f_load_path, const char* logf_load_path, double* hostA, double* hostB, double* host_f, double* host_logf, const int ID_type) {

  // Get high-precision parameters from id_params
  const bigfloat b_mu0 = static_cast<bigfloat>(id_params.mu0);
  const bigfloat b_mu1 = static_cast<bigfloat>(id_params.mu1);
  const bigfloat b_mu2 = static_cast<bigfloat>(id_params.mu2);
  const bigfloat b_beta0 = static_cast<bigfloat>(id_params.beta0);
  const bigfloat b_beta1 = static_cast<bigfloat>(id_params.beta1);
  const bigfloat b_beta2 = static_cast<bigfloat>(id_params.beta2);
  const bigfloat b_eta = static_cast<bigfloat>(id_params.eta);
  const bigfloat b_rho = static_cast<bigfloat>(id_params.rho);

  int Nmax = id_params.n_modes;

  // Deterministic start index
  int M = 10;

  // Generate initial data for f
  bigfloat* f_data = new bigfloat[2 * Nmax + 1];

  // ==== All initial data has the same f[n] profile ====
  f_data[0] = one;
  f_data[1] = one;
  host_logf[0] = 0.0; // log(1)
  host_logf[1] = 0.0;
  for (int n = 2; n < 2*Nmax+1; ++n) {
    bigfloat sum = zero;
    for (int k = 1; k <= n-1; ++k) {
      bigfloat coeff = b_mu0 + n*b_mu1 + k*b_mu2*(n-k);
      sum += coeff * f_data[k]*f_data[k]*f_data[n-k]*f_data[n-k];
    }
    f_data[n] = bmath::Sqrt(sum / (n-1));
    host_f[n] = static_cast<double>(f_data[n]);
    host_logf[n] = (f_data[n] > zero) ? static_cast<double>(blog(f_data[n])) : -1e300;
  }
  host_f[0] = static_cast<double>(f_data[0]);
  host_f[1] = static_cast<double>(f_data[1]);

  // ==== A and B initial data depend on ID_type ====
  switch(ID_type) {
    case 0: {

      // Random data subset
      for (int n = 0; n <= 5; ++n) {
        complex128 z = {gaussian_bigfloat(), two*B_PI*gaussian_bigfloat()}; // Unseeded random for production runs
        hostA[n] = static_cast<double>(z.re * bmath::cos(z.im));
        hostB[n] = static_cast<double>(z.re * bmath::sin(z.im));
      }

      // Deterministic data
      const bigfloat p = bigfloat(0.01);
      for (int n = 6; n < Nmax; ++n) {
        hostA[n] = static_cast<double>(f_data[n-5] * bmath::Pow(p, n-6));
        hostB[n] = 0.0;
      }

      break;
    }

    case 1: {
      // \alpha = random Gaussian for n < M
      for (int i = 0; i < M; ++i) {
        hostA[i] = static_cast<double>(gaussian_bigfloat());
        hostB[i] = static_cast<double>(gaussian_bigfloat());
      }

      // \alpha is random Gaussian times exponential decay for n >= M. 
      for (int i = M; i < Nmax; ++i) {
        const complex128 z = {gaussian_bigfloat(), gaussian_bigfloat()};
        const bigfloat W = bmath::Pow(static_cast<bigfloat>(i+1), b_eta/two) * bmath::exp(-b_rho*static_cast<bigfloat>(i - M));
        hostA[i] = static_cast<double>(z.re * W);
        hostB[i] = static_cast<double>(z.im * W);
      }
      
      break;
    }

    case 2: {
      // Simple power law decay for A, zero for B
      hostA[0] = static_cast<double>(one/two);
      hostB[0] = static_cast<double>(zero);
      for (int i = 1; i < Nmax; ++i) {
        hostA[i] = static_cast<double>(f_data[i] * bmath::Pow(static_cast<bigfloat>(0.5),i-1));
        hostB[i] = static_cast<double>(zero);
      }

      break;
    }

    default:
      fprintf(stderr, "Unknown ID_type %d\n", ID_type);
      delete[] f_data;
      return;
  }

  // Final NaN/Inf scans before writing files.
  // Note: host_f entries can legitimately be +Inf for large n (the profile
  // overflows double precision); the GPU kernels only ever consume
  // host_logf, so that array is the one that must stay finite.
  //debug_scan_array("host_f", host_f, 2 * Nmax + 1);
  debug_scan_array("host_logf", host_logf, 2 * Nmax + 1);
  debug_scan_array("hostA", hostA, Nmax);
  debug_scan_array("hostB", hostB, Nmax);
 
  // Save f_data, hostA, and hostB to binary files
  FILE* f_file = fopen(f_load_path, "wb");
  FILE* logf_file = fopen(logf_load_path, "wb");
  FILE* A_file = fopen(A_load_path, "wb");
  FILE* B_file = fopen(B_load_path, "wb");
  if (f_file) {
    fwrite(host_f, sizeof(double), 2*Nmax+1, f_file);
    fclose(f_file);
  } else {
    fprintf(stderr, "Failed to save f initial data\n");
  }
  if (logf_file) {
    fwrite(host_logf, sizeof(double), 2*Nmax+1, logf_file);
    fclose(logf_file);
  } else {
    fprintf(stderr, "Failed to save log(f) initial data\n");
  }
  if (A_file) {
    fwrite(hostA, sizeof(double), Nmax, A_file);
    fclose(A_file);
  } else {
    fprintf(stderr, "Failed to save A initial data\n");
  }
  if (B_file) {
    fwrite(hostB, sizeof(double), Nmax, B_file);
    fclose(B_file);
  } else {
    fprintf(stderr, "Failed to save B initial data\n");
  }

  // Memory cleanup
  delete[] f_data;

}
