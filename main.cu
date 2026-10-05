// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Anxo Biasi, Brad Cownden, Oleg Evnin, Alvaro Iturbe Jabaloyes
//
// Stochastic Cascades -- main driver: reads parameters.txt, builds the initial
// data and interaction coefficients, and runs the stochastic RK4 evolution.
// Required dependencies: CUDA toolkit (incl. cuRAND), Boost.Multiprecision.
// See README.md and LICENSE for details.

#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <sys/stat.h>
#include <stdarg.h>

#include <math.h>
#include <time.h>

#include "main.cuh"
static void append_run_log(const char* out_path, const char* fmt, ...);

// == GPU BLOCK SIZE: SET BASED ON PROBLEM SIZE AND GPU CAPABILITIES ==
#define BLOCK_SIZE       1024
#define KERNEL_TIMEOUT_WARN_MS 5000.0f
#define FINITE_DEBUG_CHECKS_DEFAULT 0

static bool finite_debug_checks_enabled()
{
	const char* env = getenv("CASCADES_FINITE_DEBUG");
	if (!env || env[0] == '\0')
	{
		return FINITE_DEBUG_CHECKS_DEFAULT != 0;
	}

	char* end = nullptr;
	long parsed = strtol(env, &end, 10);
	if (end != env)
	{
		return parsed != 0;
	}

	if (env[0] == 'f' || env[0] == 'F' || env[0] == 'n' || env[0] == 'N')
	{
		return false;
	}

	return true;
}

static bool rk_sm_enabled()
{
	const char* env = getenv("CASCADES_RK_SM");
	return env && env[0] != '\0' && env[0] != '0';
}

// CASCADES_TIMING=1 prints the average GPU time per kernel group for each
// save interval (uses the cudaEvents that are recorded anyway).
static bool timing_report_enabled()
{
	const char* env = getenv("CASCADES_TIMING");
	return env && env[0] != '\0' && env[0] != '0';
}

// Debugging utilities for checking device arrays and explaining non-finite values
static bool debug_check_device_array(const char* stage,
	const double* device_data,
	int count,
	int* device_first_non_finite,
	int outer_loop,
	int inner_loop,
	const char* out_path,
	int* out_first_non_finite)
{
	if (out_first_non_finite)
	{
		*out_first_non_finite = -1;
	}

	if (count <= 0)
	{
		return true;
	}

	init_first_non_finite<<<1, 1>>>(device_first_non_finite);
	CUDA_CHECK_KERNEL();

	const int threads = 256;
	const int blocks = (count + threads - 1) / threads;
	find_first_non_finite<<<blocks, threads>>>(device_data, count, device_first_non_finite);
	CUDA_CHECK_KERNEL();

	int first_non_finite = -1;
	CUDA_CHECK(cudaMemcpy(&first_non_finite, device_first_non_finite, sizeof(int), cudaMemcpyDeviceToHost));

	if (first_non_finite >= 0)
	{
		if (out_first_non_finite)
		{
			*out_first_non_finite = first_non_finite;
		}
		fprintf(stderr,
			"Debug: non-finite detected after %s at outer_loop=%d, inner_loop=%d, first_index=%d\n",
			stage,
			outer_loop,
			inner_loop,
			first_non_finite);
		append_run_log(out_path,
			"Debug: non-finite detected after %s at outer_loop=%d, inner_loop=%d, first_index=%d\n",
			stage,
			outer_loop,
			inner_loop,
			first_non_finite);
		return false;
	}

	printf("Debug: finite after %s at outer_loop=%d, inner_loop=%d\n", stage, outer_loop, inner_loop);
	return true;
}

// Explain the first non-finite value encountered in a deterministic manner
static void debug_explain_deterministic_non_finite(const double* deviceC,
	const double* host_f,
	int N,
	double mu0,
	double mu1,
	double mu2,
	int first_index,
	const char* out_path)
{
	if (first_index < 0)
	{
		return;
	}

	double value = 0.0;
	CUDA_CHECK(cudaMemcpy(&value, deviceC + first_index, sizeof(double), cudaMemcpyDeviceToHost));

	const int nn = N * N;
	const int m = first_index / nn;
	const int rem = first_index % nn;
	const int k = rem / N;
	const int n = rem % N;
	const int l = n + m - k;

	const bool in_triangular_domain = (m <= n);
	const bool in_k_range = (k >= m) && (k <= (n + m) / 2);
	const bool in_l_range = (l >= 0) && (l < N);
	const bool deterministic_formula_applies = (n == 0 || m == 0 || k == 0 || l == 0);
	const bool deterministic_kernel_writes = in_triangular_domain && in_k_range && in_l_range && deterministic_formula_applies;

	double denom = NAN;
	double numerator_prefactor = NAN;
	double coef_formula = NAN;
	if (n + m >= 0 && n + m <= 2 * N)
	{
		double f_nm = host_f[n + m];
		denom = f_nm * f_nm;
		numerator_prefactor = (mu2 * (n * m + k * l) + mu1 * (n + m) + mu0)
			* host_f[n] * host_f[m] * host_f[k] * ((l >= 0 && l < N) ? host_f[l] : NAN);
		coef_formula = numerator_prefactor / denom;
	}

	fprintf(stderr,
		"Debug deterministic index decode: idx=%d -> (m=%d, k=%d, n=%d, l=%d), value=%1.16e\n",
		first_index,
		m,
		k,
		n,
		l,
		value);
	fprintf(stderr,
		"Debug deterministic write-domain: m<=n=%d, k_range=%d, l_range=%d, n*m*k*l==0=%d, kernel_writes=%d\n",
		(int)in_triangular_domain,
		(int)in_k_range,
		(int)in_l_range,
		(int)deterministic_formula_applies,
		(int)deterministic_kernel_writes);
	fprintf(stderr,
		"Debug deterministic formula terms: f[n+m]^2=%1.16e, numerator_prefactor=%1.16e, coef_formula=%1.16e\n",
		denom,
		numerator_prefactor,
		coef_formula);

	append_run_log(out_path,
		"Debug deterministic index decode: idx=%d -> (m=%d, k=%d, n=%d, l=%d), value=%1.16e\n",
		first_index,
		m,
		k,
		n,
		l,
		value);
	append_run_log(out_path,
		"Debug deterministic write-domain: m<=n=%d, k_range=%d, l_range=%d, n*m*k*l==0=%d, kernel_writes=%d\n",
		(int)in_triangular_domain,
		(int)in_k_range,
		(int)in_l_range,
		(int)deterministic_formula_applies,
		(int)deterministic_kernel_writes);
	append_run_log(out_path,
		"Debug deterministic formula terms: f[n+m]^2=%1.16e, numerator_prefactor=%1.16e, coef_formula=%1.16e\n",
		denom,
		numerator_prefactor,
		coef_formula);
}

// Continue a long-running simulation by appending logs to the output file
static void append_run_log(const char* out_path, const char* fmt, ...)
{
	FILE* out = fopen(out_path, "ab");
	if (!out)
	{
		return;
	}

	va_list args;
	va_start(args, fmt);
	vfprintf(out, fmt, args);
	va_end(args);
	fclose(out);
}


// Pointer swap utility
inline void swap_double_ptr(double*& a, double*& b) {
    double* tmp = a;
    a = b;
    b = tmp;
}


// ===================================================================================
// =================================== MAIN LOOP =====================================
// ===================================================================================

int main()
{
	
	double t, J0, E0, V0, J, E;
	int i, j, loop;

	// Output files
	FILE* fsave_A, * fsave_B, * fsave_out, * fsave_t, * fsave_J, * fsave_E, * fsave_V;

	printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");

	printf("\n\tCASCADES: A first approximation to STOCHASTIC Hamiltonian systems\n");

	printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");

	// ==== PREAMBLE ====

	char parameters_txt_path[512];
	char out_path[512];
	char params_load_path[512];
	char A_load_path[512];
	char B_load_path[512];
	char f_load_path[512];
	char logf_load_path[512];

	char A_save_path[512];
	char B_save_path[512];
	char t_save_path[512];
	char J_save_path[512];
	char E_save_path[512];
	char V_save_path[512];

	sprintf(params_load_path, "%s", "parameters.txt");

	// ==== Read input parameters ====
	ID_parameters ID_params;
	evo_parameters sim_params;
	read_simulation_parameters(params_load_path, ID_params, sim_params);

	const int N = ID_params.n_modes;
	const double mu0 = ID_params.mu0;
	const double mu1 = ID_params.mu1;
	const double mu2 = ID_params.mu2;
	const double beta0 = ID_params.beta0;
	const double beta1 = ID_params.beta1;
	const double beta2 = ID_params.beta2;

	// ==== Set input/output paths ====
	build_simulation_paths(sim_params,
		parameters_txt_path, sizeof(parameters_txt_path),
		out_path, sizeof(out_path),
		A_load_path, sizeof(A_load_path),
		B_load_path, sizeof(B_load_path),
		f_load_path, sizeof(f_load_path),
		logf_load_path, sizeof(logf_load_path),
		A_save_path, sizeof(A_save_path),
		B_save_path, sizeof(B_save_path),
		t_save_path, sizeof(t_save_path),
		E_save_path, sizeof(E_save_path),
		J_save_path, sizeof(J_save_path),
		V_save_path, sizeof(V_save_path));

	const double dt = sim_params.dt;
	const int Nt_cfg = sim_params.Nt;
	const int Nsave_cfg = sim_params.Nsave;
	const int N_total_cfg = sim_params.N_total;
	const double theta_cfg = sim_params.theta;
	const double sigma_cfg = sim_params.sigma;

	// Random number seed. rng_seed = 0 in parameters.txt draws a fresh seed from
	// real entropy each run; rng_seed != 0 pins it for reproducible runs.
	// NOTE: a pinned seed makes the whole trajectory reproducible only with the
	// multi-block RK path (CASCADES_RK_SM=1). The single-block path seeds the
	// initial coefficients from run_seed but draws the per-step noise in
	// stochastic_coefficients() from clock64(), so its evolution is never
	// reproducible.
	const unsigned long long run_seed = resolve_run_seed(sim_params.rng_seed);
	const bool enable_rk_sm = rk_sm_enabled();
	const bool enable_timing_report = timing_report_enabled();

	// Off by default (FINITE_DEBUG_CHECKS_DEFAULT); set CASCADES_FINITE_DEBUG=1 to
	// scan device arrays for NaN/Inf after every kernel (slow, debugging only).
	const bool enable_finite_debug_checks = finite_debug_checks_enabled();
	const int lane_safe_n_max = BLOCK_SIZE / 2;

	// Single-kernel Runge-Kutta safety limit
	const int legacy_safe_n_max = (lane_safe_n_max < LEGACY_RK_FIXED_MAX_N) ? lane_safe_n_max : LEGACY_RK_FIXED_MAX_N;

	printf("Finite debug checks: %s (set CASCADES_FINITE_DEBUG=%d to %s)\n",
		enable_finite_debug_checks ? "ON" : "OFF",
		enable_finite_debug_checks ? 0 : 1,
		enable_finite_debug_checks ? "disable" : "enable");
  
	// Legacy single-kernel Runge-Kutta mode
	if (!enable_rk_sm) {
		printf("Legacy RK mode enabled\n");
		report_legacy_rk_bounds(BLOCK_SIZE);
		if (N > legacy_safe_n_max)
		{
			fprintf(stderr,
				"Fatal: legacy RK requires N <= min(floor(BLOCK_SIZE/2), LEGACY_RK_FIXED_MAX_N=%d). Current N=%d, BLOCK_SIZE=%d, legacy limit=%d\n", LEGACY_RK_FIXED_MAX_N, N, BLOCK_SIZE, legacy_safe_n_max);
			append_run_log(out_path, "Fatal: legacy RK limit exceeded (N=%d, BLOCK_SIZE=%d, legacy_limit=%d, fixed_max_n=%d)\n", N,
			BLOCK_SIZE,	legacy_safe_n_max, LEGACY_RK_FIXED_MAX_N);
		return 1;
		}
	} 
  
	int NUM_BLOCKS = 1;
	size_t shmem = 0;
	rk_sm_plan rk_plan = {};
	rk_sm_scratch rk_scratch;
	// Prepare the Runge-Kutta plan and scratch space if enabled
	if (enable_rk_sm && !rk_sm_make_plan(N, rk_plan))
	{
		append_run_log(out_path, "Fatal startup validation: cannot build multi-block RK launch plan for N=%d. See stderr for details.\n", N);
		return 1;
	}
	
	// Ensure the RK launch configuration is valid
  if (!validate_rk_launch_config(N, BLOCK_SIZE, NUM_BLOCKS, shmem))
	{
		append_run_log(out_path, "Fatal startup validation: invalid RK launch configuration for N=%d and BLOCK_SIZE=%d. See stderr for details.\n",	N, BLOCK_SIZE);
		return 1;
	}
	
	// ==== MEMORY ALLOCATION ====

	// Allocate host memory
	double* hostA = new double[N]; 
	double* hostB = new double[N]; 
	double* hostC = new double[N * N * N];
	double* host_f = new double[2 * N + 1];
	double* host_logf = new double[2 * N + 1];
	// Initialize host memory to zero
	std::fill(hostA, hostA + N, 0.0);
	std::fill(hostB, hostB + N, 0.0);
	std::fill(hostC, hostC + N * N * N, 0.0);
	std::fill(host_f, host_f + 2 * N + 1, 0.0);
	std::fill(host_logf, host_logf + 2 * N + 1, 0.0);

	// Create device pointers
	double* deviceA, * deviceB, * deviceC, * deviceOut_A, * deviceOut_B, * device_f, * device_logf;
	int* device_first_non_finite = nullptr;
		
	// Allocate device memory
	CUDA_CHECK(cudaMalloc(&deviceA, sizeof(double) * N));
	CUDA_CHECK(cudaMalloc(&deviceB, sizeof(double) * N));
	CUDA_CHECK(cudaMalloc(&deviceC, sizeof(double) * N * N * N));
	CUDA_CHECK(cudaMalloc(&deviceOut_A, sizeof(double) * N));
	CUDA_CHECK(cudaMalloc(&deviceOut_B, sizeof(double) * N));
	CUDA_CHECK(cudaMalloc(&device_f, sizeof(double) * (2 * N + 1)));
	CUDA_CHECK(cudaMalloc(&device_logf, sizeof(double) * (2 * N + 1)));

	// Allocate scratch space for RK if enabled
	if (enable_rk_sm)
	{
		rk_sm_alloc_scratch(rk_plan, N, rk_scratch);
	}
	if (enable_finite_debug_checks)
	{
		CUDA_CHECK(cudaMalloc(&device_first_non_finite, sizeof(int)));
	}

	// Write simulation parameters to output directory for record-keeping
	write_simulation_parameters(parameters_txt_path, ID_params, sim_params);

	// ==== LOG INITIALIZATION ====
	{
		const char* seed_source = sim_params.rng_seed != 0 ? "pinned via parameters.txt" : "auto-generated";
		const char* seed_scope = enable_rk_sm
			? "initial and per-step noise"
			: "initial coefficients only; per-step noise is clock-seeded and NOT reproducible";
		printf("RNG run_seed = %llu (%s; %s)\n", run_seed, seed_source, seed_scope);
		append_run_log(out_path, "RNG run_seed = %llu (%s; %s)\n", run_seed, seed_source, seed_scope);
	}
	printf("RK implementation: %s\n", enable_rk_sm ? "multi-block staged (CASCADES_RK_SM=1)" : "legacy single-block");
	append_run_log(out_path, "RK implementation: %s\n", enable_rk_sm ? "multi-block staged" : "legacy single-block");

	// ===============================================
	// ============= CREATE INITIAL DATA =============
	// ===============================================

	printf("Preparing initial data...\n");

	int ID_type = ID_params.ID_type;
	int data_read = ID_params.data_read;

	// ==== Read existing initial data from binary files if specified ====
	if (data_read == 0)
	{
		printf("Loading initial data from binary files...\n");
		if (!load_initial_data(A_load_path, B_load_path, f_load_path, logf_load_path, hostA, hostB, host_f, host_logf, N))
		{
			fprintf(stderr, "Failed to load initial data from binary files.\n");
			return 1;
		}
	}
	// ==== Generate initial data internally if specified ====
	else if (data_read == 1)
	{
		printf("data_read = 1. Generating initial data internally using ID_type = %d\n", ID_type);
		make_initial_data(ID_params, A_load_path, B_load_path, f_load_path, logf_load_path, hostA, hostB, host_f, host_logf, ID_type);
		printf("Wrote generated initial data to binary files for record-keeping.\n");
	}
	else
	{
		fprintf(stderr, "Invalid data_read=%d (use 0=load, 1=generate).\n", data_read);
		return 1;
	}

	printf("Done!\n");


	// ==== INITIAL DATA OVERFLOW CHECK ====

	// host_f[n] can legitimately be +Inf for large n (the profile overflows
	// double precision well before n reaches 2*N). The coefficient kernels
	// only ever consume host_logf (log(f[n])), which stays finite across the
	// whole range, so that is what must be validated here. If this still
	// trips, the f[n] recursion itself produced a non-finite/negative value
	// (e.g. from a degenerate mu0/mu1/mu2 combination), which is a genuine
	// modeling problem rather than a double-precision range issue.
	{
		int first_bad_logf = -1;
		for (int fi = 0; fi <= 2 * N; ++fi)
		{
			if (!isfinite(host_logf[fi]))
			{
				first_bad_logf = fi;
				break;
			}
		}
		if (first_bad_logf >= 0)
		{
			fprintf(stderr,
				"Fatal: host_logf[%d] = %g is non-finite (mu0=%g, mu1=%g, mu2=%g, N=%d). "
				"The f[n] profile itself is degenerate (zero/negative) at this index; "
				"check mu0/mu1/mu2 for this configuration.\n",
				first_bad_logf, host_logf[first_bad_logf], mu0, mu1, mu2, N);
			fsave_out = fopen(out_path, "ab");
			if (fsave_out)
			{
				fprintf(fsave_out,
					"Fatal: host_logf[%d] = %g is non-finite (mu0=%g, mu1=%g, mu2=%g, N=%d). "
					"The f[n] profile itself is degenerate (zero/negative) at this index; "
					"check mu0/mu1/mu2 for this configuration.\n",
					first_bad_logf, host_logf[first_bad_logf], mu0, mu1, mu2, N);
				fclose(fsave_out);
			}
			return 1;
		}
	}

	printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");

	// ==== SEND INITIAL DATA TO DEVICE ====

	CUDA_CHECK(cudaMemcpy(deviceA, hostA, sizeof(double) * N, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(deviceB, hostB, sizeof(double) * N, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(deviceC, hostC, sizeof(double) * N * N * N, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(device_f, host_f, sizeof(double) * (2 * N + 1), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(device_logf, host_logf, sizeof(double) * (2 * N + 1), cudaMemcpyHostToDevice));

	// J = sum_n |alpha_n|^2 and E = sum_n n |alpha_n|^2, with alpha_n = A_n + i B_n.
	// V is not computed: V.bin is written as zeros (one per save) as a
	// placeholder so the output file set stays fixed.
	J0 = 0.;
	E0 = 0.;
	V0 = 0.;
	for (i = 0; i < N; i++)
	{
		J0 += hostA[i] * hostA[i] + hostB[i] * hostB[i];
		E0 += i * (hostA[i] * hostA[i] + hostB[i] * hostB[i]);
	}

	t = 0.;
	loop = 0;

	printf("Initial Conserved Quantities: J0 = %1.14e, E0 = %1.14e, V0 = %1.14e\n", J0, E0, 0.);

	// Timings
	clock_t tStart = clock();
	clock_t tEnd = clock();
	clock_t tEnd0;

	// Initial data save
	fsave_A = fopen(A_save_path, "wb");
	fsave_B = fopen(B_save_path, "wb");
	fsave_t = fopen(t_save_path, "wb");
	fsave_J = fopen(J_save_path, "wb");
	fsave_E = fopen(E_save_path, "wb");
	fsave_V = fopen(V_save_path, "wb");

	fwrite(hostA, sizeof(double), N, fsave_A);
	fwrite(hostB, sizeof(double), N, fsave_B);
	fwrite(&t, sizeof(double), 1, fsave_t);
	fwrite(&J0, sizeof(double), 1, fsave_J);
	fwrite(&E0, sizeof(double), 1, fsave_E);
	fwrite(&V0, sizeof(double), 1, fsave_V);

	// ==== INITIALIZATION COMPLETE ====

	printf("Initialization  >>>>>>  loop = %i, t = %f, J = %1.14e, E = %1.14e, Nsave = %i\n", loop, t, J0, E0, Nsave_cfg);
	fsave_out = fopen(out_path, "ab");
	fprintf(fsave_out, "Initialization  >>>>>>  loop = %i, t = %f, J = %1.14e, E = %1.14e, Nsave = %i\n", loop, t, J0, E0, Nsave_cfg);
	fclose(fsave_out);

	// ==== BUILD DETERMINISTIC COEFFICIENTS ====

	int coeff_blocks = (N * N * N + BLOCK_SIZE - 1) / BLOCK_SIZE;
	build_deterministic_coefficients<<<coeff_blocks, BLOCK_SIZE>>>(device_logf, deviceC, N, mu0, mu1, mu2);
	#ifdef DEBUG
	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK_KERNEL();
	#endif

	if (enable_finite_debug_checks)
	{
		int deterministic_bad_index = -1;
		if (!debug_check_device_array("build_deterministic_coefficients", deviceC, N * N * N, device_first_non_finite, -1, -1, out_path, &deterministic_bad_index))
		{
			debug_explain_deterministic_non_finite(deviceC, host_f, N, mu0, mu1, mu2, deterministic_bad_index, out_path);
			return 1;
		}
	}

	// ==== DEVICE GEOMETRY ====

	// Block sizing for kernel launches
	unsigned int block_x = (N > 32) ? 32u : (unsigned int)(N - 1);
	unsigned int block_y = (N > 8) ? 8u : 1u;
	dim3 block(block_x, block_y);
	dim3 grid((N - 1 + block.x - 1) / block.x, (N - 1 + block.y - 1) / block.y);

	unsigned int odd_block_x = (N > 32) ? 32u : (unsigned int)(N - 1);
	unsigned int odd_block_y = (N > 8) ? 8u : 1u;
	dim3 odd_block(odd_block_x, odd_block_y);
	dim3 odd_grid((N - 1 + odd_block.x - 1) / odd_block.x, (N - 1 + odd_block.y - 1) / odd_block.y);

	unsigned int even_block_x = (N > 32) ? 32u : (unsigned int)(N - 1);
	unsigned int even_block_y = (N > 8) ? 8u : 1u;
	dim3 even_block(even_block_x, even_block_y);
	dim3 even_grid((N - 1 + even_block.x - 1) / even_block.x, (N - 1 + even_block.y - 1) / even_block.y);

	// Initialize stochastic coefficients from the OU stationary distribution
	initialize_stochastic_coefficients_sm<<<grid, block>>>(deviceC, run_seed, sigma_cfg, N);
	CUDA_CHECK_KERNEL();

	// Device timing events for kernels
	cudaEvent_t start, stop_stoch, stop_odd, stop_even, stop_rk;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop_stoch));
	CUDA_CHECK(cudaEventCreate(&stop_odd));
	CUDA_CHECK(cudaEventCreate(&stop_even));
	CUDA_CHECK(cudaEventCreate(&stop_rk));

	// ==== STOCHASTIC RK EVOLUTION ====
	for (i = 0; i < N_total_cfg; i++)
	{
		int last_inner_loop = -1;
		float ms_stoch = 0.0f, ms_odd = 0.0f, ms_even = 0.0f, ms_rk = 0.0f;

		for (j = 0; j < Nsave_cfg; j++)
		{
			last_inner_loop = j;
			float milliseconds = 0.0f;

			// Seed for repeatable stochastic updates
			int step_seed = i * Nsave_cfg + j;

			// === Stochastic coefficients calculation ===
			CUDA_CHECK(cudaEventRecord(start));
			if(enable_rk_sm) {
				stochastic_coefficients_sm<<<grid, block>>>(deviceC, run_seed, step_seed, theta_cfg, sigma_cfg, dt, N);
			} else {
				stochastic_coefficients<<<grid, block>>>(deviceC, step_seed, theta_cfg, sigma_cfg, dt, N);
			}

			#ifdef DEBUG
			CUDA_CHECK(cudaDeviceSynchronize());
			CUDA_CHECK(cudaMemcpy(hostC, deviceC, sizeof(double) * N * N * N, cudaMemcpyDeviceToHost));
			printf("Sample stochastic coefficient after kernel: C[0] = %1.5e\n", hostC[1*N*N+ 1*N+1]);
			printf("Sample stochastic coefficient after kernel: C[last] = %1.5e\n", hostC[(N-1) * (N * N + N + 1)]);
			if (enable_finite_debug_checks &&
				!debug_check_device_array("stochastic_coefficients", deviceC, N * N * N, device_first_non_finite, i, j, out_path, nullptr))
			{
				return 1;
			}
			#endif

			// Stochastic timing
			CUDA_CHECK(cudaEventRecord(stop_stoch));

			// ==== (n+ m) IS ODD CORRECTIONS ON DEVICE ====
			apply_odd_corrections<<<odd_grid, odd_block>>>(device_logf, deviceC, N, beta0, beta1, beta2);
			if (enable_finite_debug_checks &&
				!debug_check_device_array("apply_odd_corrections", deviceC, N * N * N, device_first_non_finite, i, j, out_path, nullptr))
			{
				return 1;
			}
			CUDA_CHECK(cudaEventRecord(stop_odd));

			// ==== (n + m) IS EVEN CORRECTIONS ON DEVICE ====
			apply_even_corrections<<<even_grid, even_block>>>(device_logf, deviceC, N, beta0, beta1, beta2);
			if (enable_finite_debug_checks &&
				!debug_check_device_array("apply_even_corrections", deviceC, N * N * N, device_first_non_finite, i, j, out_path, nullptr))
			{
				return 1;
			}
			CUDA_CHECK(cudaEventRecord(stop_even));

			// ==== RK STEP ON DEVICE WITH STOCHASTIC EVOLUTION ====
			// The opt-in multi-block path uses launch boundaries as
			// grid-wide barriers between the four RK stages.
			if (enable_rk_sm)
			{
				rk_sm_step(rk_plan, deviceA, deviceB, deviceC, rk_scratch, dt, N, Nt_cfg);
			}
			else
			{
				RK_step <<<NUM_BLOCKS, BLOCK_SIZE>>> (deviceA, deviceB, deviceC, dt, deviceOut_A, deviceOut_B, N, Nt_cfg);
			}

			double* rk_check_A = enable_rk_sm ? deviceA : deviceOut_A;
			double* rk_check_B = enable_rk_sm ? deviceB : deviceOut_B;
			if (enable_finite_debug_checks &&
				!debug_check_device_array("RK_step (out_A)", rk_check_A, N, device_first_non_finite, i, j, out_path, nullptr))
			{
				return 1;
			}
			if (enable_finite_debug_checks &&
				!debug_check_device_array("RK_step (out_B)", rk_check_B, N, device_first_non_finite, i, j, out_path, nullptr))
			{
				return 1;
			}

			CUDA_CHECK(cudaEventRecord(stop_rk));

			// Single blocking sync per iteration (events complete in stream
			// issue order, so this also guarantees stop_stoch/odd/even are
			// ready) instead of one sync per kernel -- cuts host/device
			// round-trips 4x without losing per-kernel timing/warnings.
			CUDA_CHECK(cudaEventSynchronize(stop_rk));

			CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop_stoch));
			ms_stoch += milliseconds;
			if (milliseconds > KERNEL_TIMEOUT_WARN_MS)
			{
				fprintf(stderr, "Warning: stochastic_coefficients took %.3f ms at cfg=(i=%d, j=%d), N=%d\n", milliseconds, i, j, N);
				append_run_log(out_path, "Warning: stochastic_coefficients took %.3f ms at cfg=(i=%d, j=%d), N=%d\n", milliseconds, i, j, N);
			}
			#ifdef DEBUG
			printf("Stochastic coefficients calculation time: %f ms\n", milliseconds);
			#endif

			CUDA_CHECK(cudaEventElapsedTime(&milliseconds, stop_stoch, stop_odd));
			ms_odd += milliseconds;
			if (milliseconds > KERNEL_TIMEOUT_WARN_MS)
			{
				fprintf(stderr, "Warning: apply_odd_corrections took %.3f ms at cfg=(i=%d, j=%d), N=%d\n", milliseconds, i, j, N);
				append_run_log(out_path, "Warning: apply_odd_corrections took %.3f ms at cfg=(i=%d, j=%d), N=%d\n", milliseconds, i, j, N);
			}
			#ifdef DEBUG
			printf("Odd corrections calculation time: %f ms\n", milliseconds);
			#endif

			CUDA_CHECK(cudaEventElapsedTime(&milliseconds, stop_odd, stop_even));
			ms_even += milliseconds;
			if (milliseconds > KERNEL_TIMEOUT_WARN_MS)
			{
				fprintf(stderr, "Warning: apply_even_corrections took %.3f ms at cfg=(i=%d, j=%d), N=%d\n", milliseconds, i, j, N);
				append_run_log(out_path, "Warning: apply_even_corrections took %.3f ms at cfg=(i=%d, j=%d), N=%d\n", milliseconds, i, j, N);
			}
			#ifdef DEBUG
			printf("Even corrections calculation time: %f ms\n", milliseconds);
			#endif

			CUDA_CHECK(cudaEventElapsedTime(&milliseconds, stop_even, stop_rk));
			ms_rk += milliseconds;
			if (milliseconds > KERNEL_TIMEOUT_WARN_MS)
			{
				fprintf(stderr, "Warning: RK_step took %.3f ms at cfg=(i=%d, j=%d), N=%d, Nt=%d\n", milliseconds, i, j, N, Nt_cfg);
				append_run_log(out_path, "Warning: RK_step took %.3f ms at cfg=(i=%d, j=%d), N=%d, Nt=%d\n", milliseconds, i, j, N, Nt_cfg);
			}
			#ifdef DEBUG
			printf("RK step calculation time: %f ms\n", milliseconds);
			#endif

			// ==== LEGACY RK: OUTPUT FROM RK BECOMES INPUT FOR NEXT STEP ====
			if (!enable_rk_sm)
			{
				swap_double_ptr(deviceA, deviceOut_A);
				swap_double_ptr(deviceB, deviceOut_B);
			}
		}

		// ==== CALCULATE EVOLUTION DIAGNOSTICS ON HOST ====
		CUDA_CHECK(cudaMemcpy(hostA, deviceA, sizeof(double) * N, cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(hostB, deviceB, sizeof(double) * N, cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaDeviceSynchronize());

		J = 0.;
		E = 0.;
		int first_non_finite = -1;
		for (j = 0; j < N; j++)
		{
			if (first_non_finite < 0 && (!isfinite(hostA[j]) || !isfinite(hostB[j])))
			{
				first_non_finite = j;
			}
			J += hostA[j] * hostA[j] + hostB[j] * hostB[j];
			E += j * (hostA[j] * hostA[j] + hostB[j] * hostB[j]);
		}

		// ==== CHECK FOR NON-FINITE VALUES IN DIAGNOSTICS ====
		if (first_non_finite >= 0 || !isfinite(J) || !isfinite(E))
		{
			fprintf(stderr,
				"Fatal: non-finite state detected after RK integration at outer_loop=%d, inner_loop=%d, mode_index=%d, N=%d, BLOCK_SIZE=%d\n",
				i,
				last_inner_loop,
				first_non_finite,
				N,
				BLOCK_SIZE);
			append_run_log(out_path,
				"Fatal: non-finite state detected after RK integration at outer_loop=%d, inner_loop=%d, mode_index=%d, N=%d, BLOCK_SIZE=%d\n",
				i,
				last_inner_loop,
				first_non_finite,
				N,
				BLOCK_SIZE);
			fclose(fsave_A);
			fclose(fsave_B);
			fclose(fsave_t);
			fclose(fsave_J);
			fclose(fsave_E);
			fclose(fsave_V);
			return 1;
		}


		// ==== SAVE STATE TO OUTPUT FILES ====
		t = double(i + 1) * double(Nsave_cfg) * double(Nt_cfg) * dt;

		fwrite(hostA, sizeof(double), N, fsave_A);
		fwrite(hostB, sizeof(double), N, fsave_B);
		fwrite(&t, sizeof(double), 1, fsave_t);
		fwrite(&J, sizeof(double), 1, fsave_J);
		fwrite(&E, sizeof(double), 1, fsave_E);
		fwrite(&V0, sizeof(double), 1, fsave_V);

		tEnd0 = tEnd;
		tEnd = clock();

		loop += 1;

		// ==== REPORT TIMING AND LOOP INFORMATION ====
		printf("Total time: %.2fs,  time: %.2fs,  loop = %i,  t = %f,  Delta_{rel} J = %E,  Delta_{rel} E = %E,  Nsave = %i \n", (double)(tEnd - tStart) / (CLOCKS_PER_SEC), (double)(tEnd - tEnd0) / CLOCKS_PER_SEC, loop, t, (J - J0) / J0, (E - E0) / E0, Nsave_cfg);
		if (enable_timing_report && Nsave_cfg > 0)
		{
			printf("  GPU ms/step (avg over %d steps): stochastic %.3f, odd %.3f, even %.3f, RK %.3f, total %.3f\n",
				Nsave_cfg, ms_stoch / Nsave_cfg, ms_odd / Nsave_cfg, ms_even / Nsave_cfg, ms_rk / Nsave_cfg,
				(ms_stoch + ms_odd + ms_even + ms_rk) / Nsave_cfg);
		}
		fsave_out = fopen(out_path, "ab");
		fprintf(fsave_out, "Total time: %.2fs,  time: %.2fs,  loop = %i,  t = %f,  Delta_{rel} J = %E,  Delta_{rel} E = %E,  Nsave = %i \n", (double)(tEnd - tStart) / (CLOCKS_PER_SEC), (double)(tEnd - tEnd0) / CLOCKS_PER_SEC, loop, t, (J - J0) / J0, (E - E0) / E0, Nsave_cfg);
		fclose(fsave_out);
	
	} // End outer loop

	printf("Done!\n");
	printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");

	// ==== CLOSE OUTPUT FILES ====
	fclose(fsave_A);
	fclose(fsave_B);
	fclose(fsave_t);
	fclose(fsave_J);
	fclose(fsave_E);
	fclose(fsave_V);

	// ==== MEMORY CLEANUP ====
	delete[] hostA;
	delete[] hostB;
	delete[] hostC;
	delete[] host_f;
	delete[] host_logf;
	CUDA_CHECK(cudaFree(deviceA));
	CUDA_CHECK(cudaFree(deviceB));
	CUDA_CHECK(cudaFree(deviceC));
	CUDA_CHECK(cudaFree(deviceOut_A));
	CUDA_CHECK(cudaFree(deviceOut_B));
	CUDA_CHECK(cudaFree(device_f));
	CUDA_CHECK(cudaFree(device_logf));
	if (enable_rk_sm)
	{
		rk_sm_free_scratch(rk_scratch);
	}
	if (enable_finite_debug_checks)
	{
		CUDA_CHECK(cudaFree(device_first_non_finite));
	}
	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop_stoch));
	CUDA_CHECK(cudaEventDestroy(stop_odd));
	CUDA_CHECK(cudaEventDestroy(stop_even));
	CUDA_CHECK(cudaEventDestroy(stop_rk));

} //end main
