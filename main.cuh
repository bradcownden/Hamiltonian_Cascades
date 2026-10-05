// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Anxo Biasi, Brad Cownden, Oleg Evnin, Alvaro Iturbe Jabaloyes
//
// Shared declarations: parameter structs, CUDA error-check macros,
// kernel prototypes and launch-plan types.
// See README.md and LICENSE for details.

#pragma once

#include <string.h>
#include <stdio.h>
#include <cuda_runtime.h>

// Debug on
/* 
#ifndef DEBUG
#define DEBUG 0
#endif  */

// Debug off

#ifdef DEBUG
#undef DEBUG
#endif


// Abort on any CUDA error, printing file/line for easy diagnosis.
#define CUDA_CHECK(call) \
	do { \
		cudaError_t _e = (call); \
		if (_e != cudaSuccess) { \
			fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
			exit(1); \
		} \
	} while (0)

// Check for errors from the most recent kernel launch.
#define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())

struct ID_parameters
{
  int n_modes;
	double mu0;
	double mu1;
	double mu2;
	double beta0;
	double beta1;
	double beta2;
	double eta;
	double rho;
	int data_read; // 0: load from binary files, 1: generate initial data
	int ID_type;
};

struct evo_parameters
{
	int Nt;
	int Nsave;
	int N_total;	
	double dt;
	double theta;
	double sigma;
	unsigned long long rng_seed; // 0 = draw a fresh non-deterministic seed at startup; nonzero = reproducible run
	char data_in[256];
	char data_out[256];
};

static inline void trim(char* s) {
    char* p = s;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    memmove(s, p, strlen(p) + 1);

    size_t len = strlen(s);
    while (len > 0 && (s[len-1] == ' ' || s[len-1] == '\t' || s[len-1] == '\n' || s[len-1] == '\r')) {
        s[len-1] = '\0';
        len--;
    }
}

__device__ double RK_coef[3] = { 0.5, 0.5, 1. };

constexpr int LEGACY_RK_FIXED_MAX_N = 512;
constexpr int LEGACY_RK_SHARED_ARRAY_COUNT = 10;
constexpr size_t LEGACY_RK_STATIC_SHMEM_BYTES =
	(size_t)LEGACY_RK_FIXED_MAX_N * (size_t)LEGACY_RK_SHARED_ARRAY_COUNT * sizeof(double);

bool load_initial_data(const char* a_filename, const char* b_filename, const char* f_filename, const char* logf_filename, double* A, double* B, double* f, double* logf, int n);

void build_simulation_paths(const evo_parameters& sim_params,
	char* parameters_txt_path, size_t parameters_txt_path_size,
	char* out_path, size_t out_path_size,
	char* A_load_path, size_t A_load_path_size,
	char* B_load_path, size_t B_load_path_size,
	char* f_load_path, size_t f_load_path_size,
	char* logf_load_path, size_t logf_load_path_size,
	char* A_save_path, size_t A_save_path_size,
	char* B_save_path, size_t B_save_path_size,
	char* t_save_path, size_t t_save_path_size,
	char* E_save_path, size_t E_save_path_size,
	char* J_save_path, size_t J_save_path_size,
	char* V_save_path, size_t V_save_path_size);

bool validate_rk_hardware_limits(int N, int block_size, size_t& shmem);
bool validate_rk_launch_config(int N, int block_size, int& num_blocks, size_t& shmem);
void report_legacy_rk_bounds(int block_size);

void read_simulation_parameters(const char* filename, ID_parameters& id_params, evo_parameters& evo_params);
void write_simulation_parameters(const char* filename, const ID_parameters& id_params, const evo_parameters& evo_params);


void make_initial_data(const ID_parameters& id_params, const char* A_load_path, const char* B_load_path, const char* f_load_path, const char* logf_load_path, double* hostA, double* hostB, double* host_f, double* host_logf, const int ID_type);

__host__ __device__ double KroneckerDelta(int a, int b);

__device__ double uniform_random_from_index(int n, int m, int k, int step_seed);
__device__ double random_coef_from_index(int n, int m, int k, int step_seed);
__device__ double gaussian_random_from_index(int n, int m, int k);

__global__ void build_deterministic_coefficients(const double* logf, double* C, int N, double mu0, double mu1, double mu2);
__global__ void build_random_coefficients(double* C, int step_seed, int N);
__global__ void stochastic_coefficients(double* C, int step_seed, double theta, double sigma, double dt, int N);
__global__ void initialize_stochastic_coefficients(double* C, int N);
__global__ void apply_odd_corrections(const double* logf, double* C, int N, double b0, double b1, double b2);
__global__ void apply_even_corrections(const double* logf, double* C, int N, double b0, double b1, double b2);
__global__ void init_first_non_finite(int* first_idx);
__global__ void find_first_non_finite(const double* data, int count, int* first_idx);

__global__ void RK_step(double* A, double* B, double* C, double dt, double* outA, double* outB, int N, int Nt);

__global__ void stochastic_coefficients_sm(double* C, unsigned long long run_seed, int step_seed, double theta, double sigma, double dt, int N);

__global__ void initialize_stochastic_coefficients_sm(double* C, unsigned long long run_seed, double sigma, int N);

// ==== Multi-block RK4 (rk_sm.cu), opt-in via CASCADES_RK_SM=1 ====
constexpr int RK_SM_THREADS = 64;      // modes per block (1-D); small so the grid can spread over all SMs
constexpr int RK_SM_BLOCKS_PER_SM = 8; // target grid size = this * SM count

struct rk_sm_plan
{
	int threads;   // == RK_SM_THREADS
	int l_blocks;  // ceil(N / threads)
	int k_splits;  // number of K chunks (gridDim.y of rk_sm_partial)
	int k_chunk;   // K values per chunk
	size_t shmem;  // dynamic shared memory per block (2*N doubles)
};

struct rk_sm_scratch
{
	double* tmpA = nullptr; double* tmpB = nullptr; // stage ping-pong buffer
	double* A0 = nullptr;   double* B0 = nullptr;   // state at start of step
	double* k1A = nullptr;  double* k1B = nullptr;
	double* k2A = nullptr;  double* k2B = nullptr;
	double* k3A = nullptr;  double* k3B = nullptr;
	double* partA = nullptr; double* partB = nullptr; // [k_splits][N] partial sums
};

__global__ void rk_sm_partial(const double* A, const double* B, const double* C,
	double* partA, double* partB, int N, int k_chunk);
__global__ void rk_sm_combine(const double* partA, const double* partB, int k_splits,
	const double* Ain, const double* Bin, double* A0, double* B0, double* Aout, double* Bout,
	double* k1A, double* k1B, double* k2A, double* k2B, double* k3A, double* k3B,
	double dt, int N, int stage);
bool rk_sm_make_plan(int N, rk_sm_plan& plan);
void rk_sm_alloc_scratch(const rk_sm_plan& plan, int N, rk_sm_scratch& s);
void rk_sm_free_scratch(rk_sm_scratch& s);
void rk_sm_step(const rk_sm_plan& plan, double* A, double* B, const double* C, rk_sm_scratch& s, double dt, int N, int Nt);

// ==== Seeded stochastic coefficients (rk_sm.cu) ====
// Draws a fresh non-deterministic 64-bit seed unless the caller has pinned
// one via parameters.txt (rng_seed != 0), for reproducible regression runs.
unsigned long long resolve_run_seed(unsigned long long configured_seed);
__device__ double gaussian_random_from_seed(unsigned long long run_seed, int step_seed, int n, int m, int k);
__global__ void initialize_stochastic_coefficients_v3(const double* logf, double* C, unsigned long long run_seed, int N, double mu0, double mu1, double mu2);
__global__ void stochastic_coefficients_v3(const double* logf, double* C, unsigned long long run_seed, int step_seed, double theta, double sigma, double dt, int N, double mu0, double mu1, double mu2);
