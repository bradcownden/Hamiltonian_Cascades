// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Anxo Biasi, Brad Cownden, Oleg Evnin, Alvaro Iturbe Jabaloyes
//
// Input/output: parameters.txt parsing, output paths, binary initial-data
// loading and RK launch-configuration validation.
// See README.md and LICENSE for details.

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include "main.cuh"
#include <filesystem>

void set_output_dir(evo_parameters& evo_params);

static bool read_binary_array(const char* filename, double* target, size_t count, const char* label)
{
	FILE* fp = fopen(filename, "rb");
	if (!fp)
	{
		fprintf(stderr, "Error: cannot open %s file %s\n", label, filename);
		return false;
	}

	size_t read_count = fread(target, sizeof(double), count, fp);
	fclose(fp);

	if (read_count != count)
	{
		fprintf(stderr, "Error: %s file %s has %zu values, expected %zu\n", label, filename, read_count, count);
		return false;
	}

	return true;
}

bool load_initial_data(const char* a_filename,
	const char* b_filename,
	const char* f_filename,
	const char* logf_filename,
	double* A,
	double* B,
	double* f,
	double* logf,
	int n)
{
	if (!read_binary_array(a_filename, A, (size_t)n, "A"))
	{
		return false;
	}
	if (!read_binary_array(b_filename, B, (size_t)n, "B"))
	{
		return false;
	}
	if (!read_binary_array(f_filename, f, (size_t)(2 * n + 1), "f"))
	{
		return false;
	}
	if (!read_binary_array(logf_filename, logf, (size_t)(2 * n + 1), "logf"))
	{
		return false;
	}

	return true;
}

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
	char* V_save_path, size_t V_save_path_size)
{
	snprintf(parameters_txt_path, parameters_txt_path_size, "./%s/simulation_parameters.txt", sim_params.data_out);
	snprintf(out_path, out_path_size, "./%s/out.txt", sim_params.data_out);
	snprintf(A_load_path, A_load_path_size, "./%s/%s", sim_params.data_in, "fload_A.bin");
	snprintf(B_load_path, B_load_path_size, "./%s/%s", sim_params.data_in, "fload_B.bin");
	snprintf(f_load_path, f_load_path_size, "./%s/%s", sim_params.data_in, "fload_f.bin");
	snprintf(logf_load_path, logf_load_path_size, "./%s/%s", sim_params.data_in, "fload_logf.bin");
	snprintf(A_save_path, A_save_path_size, "./%s/%s", sim_params.data_out, "A.bin");
	snprintf(B_save_path, B_save_path_size, "./%s/%s", sim_params.data_out, "B.bin");
	snprintf(t_save_path, t_save_path_size, "./%s/%s", sim_params.data_out, "t.bin");
	snprintf(E_save_path, E_save_path_size, "./%s/%s", sim_params.data_out, "E.bin");
	snprintf(J_save_path, J_save_path_size, "./%s/%s", sim_params.data_out, "J.bin");
	snprintf(V_save_path, V_save_path_size, "./%s/%s", sim_params.data_out, "V.bin");
}

bool validate_rk_hardware_limits(int N, int block_size, size_t& shmem)
{
	if (N <= 0)
	{
		fprintf(stderr, "Error: N must be positive. Current N = %d\n", N);
		return false;
	}

	if (block_size <= 0)
	{
		fprintf(stderr, "Error: BLOCK_SIZE must be positive. Current BLOCK_SIZE = %d\n", block_size);
		return false;
	}

	int device_id = 0;
	if (cudaGetDevice(&device_id) != cudaSuccess)
	{
		fprintf(stderr, "Error: cannot query current CUDA device.\n");
		return false;
	}

	int max_threads_per_block = 0;
	if (cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device_id) != cudaSuccess)
	{
		fprintf(stderr, "Error: cannot query cudaDevAttrMaxThreadsPerBlock.\n");
		return false;
	}
	if (block_size > max_threads_per_block)
	{
		fprintf(stderr, "Error: BLOCK_SIZE = %d exceeds device max threads per block = %d\n", block_size, max_threads_per_block);
		return false;
	}

	int max_shmem_default = 0;
	if (cudaDeviceGetAttribute(&max_shmem_default, cudaDevAttrMaxSharedMemoryPerBlock, device_id) != cudaSuccess)
	{
		fprintf(stderr, "Error: cannot query cudaDevAttrMaxSharedMemoryPerBlock.\n");
		return false;
	}
	int max_shmem_optin = 0;
	if (cudaDeviceGetAttribute(&max_shmem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id) != cudaSuccess)
	{
		fprintf(stderr, "Error: cannot query cudaDevAttrMaxSharedMemoryPerBlockOptin.\n");
		return false;
	}

	printf("RK launch capability: device maxThreadsPerBlock=%d, optInSharedMemPerBlock=%d B, BLOCK_SIZE=%d, requested N=%d\n",
		max_threads_per_block, max_shmem_optin, block_size, N);

	shmem = 0;

	#ifdef DEBUG
	cudaDeviceProp prop;
	if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess)
	{
		printf("Shared memory avialable per block: %zu bytes\n", prop.sharedMemPerBlock);
	}
	cudaFuncAttributes attr;
	if (cudaFuncGetAttributes(&attr, RK_step) == cudaSuccess)
	{
		printf("Static shared memory: %zu bytes\n", attr.sharedSizeBytes);
	}
	#endif

	return true;
}

bool validate_rk_launch_config(int N, int block_size, int& num_blocks, size_t& shmem)
{
	if (!validate_rk_hardware_limits(N, block_size, shmem))
	{
		return false;
	}

	num_blocks = ((N + block_size - 1) / block_size);
	if (num_blocks != 1)
	{
		fprintf(stderr, "Error: RK_step currently supports exactly one block. Use N <= BLOCK_SIZE. Current N = %d, BLOCK_SIZE = %d\n", N, block_size);
		return false;
	}

	const int lane_safe_n_max = block_size / 2;
	const int legacy_safe_n_max = (lane_safe_n_max < LEGACY_RK_FIXED_MAX_N) ? lane_safe_n_max : LEGACY_RK_FIXED_MAX_N;
	if (N > legacy_safe_n_max)
	{
		fprintf(stderr,
			"Fatal: legacy RK requires N <= min(floor(BLOCK_SIZE/2), LEGACY_RK_FIXED_MAX_N=%d). "
			"Current N = %d, BLOCK_SIZE = %d, safe max N = %d.\n",
			LEGACY_RK_FIXED_MAX_N,
			N,
			block_size,
			legacy_safe_n_max);
		return false;
	}

	if (N > (int)(0.9 * (double)legacy_safe_n_max))
	{
		fprintf(stderr,
			"Warning: N = %d is near the legacy RK safety limit floor(BLOCK_SIZE/2) = %d.\n",
			N,
			legacy_safe_n_max);
	}

	int device_id = 0;
	if (cudaGetDevice(&device_id) != cudaSuccess)
	{
		fprintf(stderr, "Error: cannot query current CUDA device.\n");
		return false;
	}

	int max_shmem_optin = 0;
	if (cudaDeviceGetAttribute(&max_shmem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id) != cudaSuccess)
	{
		fprintf(stderr, "Error: cannot query cudaDevAttrMaxSharedMemoryPerBlockOptin.\n");
		return false;
	}

	int max_threads_per_block = 0;
	if (cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device_id) != cudaSuccess)
	{
		fprintf(stderr, "Error: cannot query cudaDevAttrMaxThreadsPerBlock.\n");
		return false;
	}

	int warp_size = 32;
	if (cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, device_id) != cudaSuccess)
	{
		warp_size = 32;
	}

	int max_regs_per_block = 0;
	if (cudaDeviceGetAttribute(&max_regs_per_block, cudaDevAttrMaxRegistersPerBlock, device_id) != cudaSuccess)
	{
		fprintf(stderr, "Error: cannot query cudaDevAttrMaxRegistersPerBlock.\n");
		return false;
	}

	shmem = 0;
	if (LEGACY_RK_STATIC_SHMEM_BYTES > (size_t)max_shmem_optin)
	{
		fprintf(stderr,
			"Error: legacy RK static shared memory (%zu bytes) exceeds device opt-in limit (%d bytes). "
			"Reduce LEGACY_RK_FIXED_MAX_N or use dynamic-shared kernel.\n",
			(size_t)LEGACY_RK_STATIC_SHMEM_BYTES,
			max_shmem_optin);
		return false;
	}

	cudaFuncAttributes rk_attr{};
	if (cudaFuncGetAttributes(&rk_attr, RK_step) != cudaSuccess)
	{
		fprintf(stderr, "Error: cannot query RK_step function attributes.\n");
		return false;
	}

	int active_blocks = 0;
	cudaError_t occupancy_status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, RK_step, block_size, 0);
	if (occupancy_status != cudaSuccess || active_blocks < 1)
	{
		const char* occupancy_reason = nullptr;
		if (occupancy_status != cudaSuccess)
		{
			occupancy_reason = cudaGetErrorString(occupancy_status);
		}
		else
		{
			occupancy_reason = "active blocks per SM is 0 (insufficient launch resources)";
		}

		int max_threads_hw = max_threads_per_block;
		if (rk_attr.maxThreadsPerBlock > 0 && rk_attr.maxThreadsPerBlock < max_threads_hw)
		{
			max_threads_hw = rk_attr.maxThreadsPerBlock;
		}

		int regs_per_thread = rk_attr.numRegs > 0 ? rk_attr.numRegs : 1;
		int max_threads_regs = (max_regs_per_block / regs_per_thread);
		if (warp_size > 0)
		{
			max_threads_regs = (max_threads_regs / warp_size) * warp_size;
		}
		if (max_threads_regs < 0)
		{
			max_threads_regs = 0;
		}

		int max_threads_launchable = max_threads_hw;
		if (max_threads_regs > 0 && max_threads_regs < max_threads_launchable)
		{
			max_threads_launchable = max_threads_regs;
		}

		int max_n_lane_from_launchable = max_threads_launchable / 2;
		int max_n_effective = max_n_lane_from_launchable;
		if (LEGACY_RK_FIXED_MAX_N < max_n_effective)
		{
			max_n_effective = LEGACY_RK_FIXED_MAX_N;
		}

		int max_block_size_for_kernel = 0;
		for (int candidate = max_threads_hw; candidate >= warp_size; candidate -= warp_size)
		{
			int candidate_active_blocks = 0;
			cudaError_t candidate_status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&candidate_active_blocks, RK_step, candidate, 0);
			if (candidate_status == cudaSuccess && candidate_active_blocks > 0)
			{
				max_block_size_for_kernel = candidate;
				break;
			}
		}

		fprintf(stderr,
			"Error: RK_step launch is not feasible for BLOCK_SIZE=%d, N=%d, static_shmem=%zu bytes (%s).\n",
			block_size,
			N,
			(size_t)rk_attr.sharedSizeBytes,
			occupancy_reason);
		fprintf(stderr,
			"Max hardware resources: maxThreadsPerBlock=%d, maxRegistersPerBlock=%d, maxOptInSharedMemPerBlock=%d bytes.\n",
			max_threads_per_block,
			max_regs_per_block,
			max_shmem_optin);
		fprintf(stderr,
			"RK_step attributes: numRegsPerThread=%d, staticSharedMem=%zu bytes, kernelMaxThreadsPerBlock=%d.\n",
			rk_attr.numRegs,
			(size_t)rk_attr.sharedSizeBytes,
			rk_attr.maxThreadsPerBlock);
		fprintf(stderr,
			"Estimated feasible limits: maxBlockSizeByRegisters=%d, maxBlockSizeByKernelAndHW=%d, staticRKSharedMem=%zu bytes.\n",
			max_threads_regs,
			max_threads_hw,
			(size_t)rk_attr.sharedSizeBytes);
		fprintf(stderr,
			"Estimated legacy N limit from resources: N <= min(maxBlockSize/2=%d, fixed_buffer_limit=%d) = %d.\n",
			max_n_lane_from_launchable,
			LEGACY_RK_FIXED_MAX_N,
			max_n_effective);
		if (max_block_size_for_kernel > 0)
		{
			int candidate_n_limit = max_block_size_for_kernel / 2;
			if (LEGACY_RK_FIXED_MAX_N < candidate_n_limit)
			{
				candidate_n_limit = LEGACY_RK_FIXED_MAX_N;
			}
			fprintf(stderr,
				"For current N=%d, largest launchable RK block size appears to be %d (legacy N <= min(%d/2, %d) = %d).\n",
				N,
				max_block_size_for_kernel,
				max_block_size_for_kernel,
				LEGACY_RK_FIXED_MAX_N,
				candidate_n_limit);
		}
		return false;
	}

	printf("Legacy RK safety limit: N <= %d for BLOCK_SIZE=%d\n", legacy_safe_n_max, block_size);

	return true;
}

void report_legacy_rk_bounds(int block_size)
{
	if (block_size <= 0)
	{
		fprintf(stderr, "Legacy RK bounds: invalid BLOCK_SIZE=%d\n", block_size);
		return;
	}

	int device_id = 0;
	if (cudaGetDevice(&device_id) != cudaSuccess)
	{
		fprintf(stderr, "Legacy RK bounds: cannot query CUDA device.\n");
		return;
	}

	int max_threads_per_block = 0;
	if (cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device_id) != cudaSuccess)
	{
		fprintf(stderr, "Legacy RK bounds: cannot query max threads per block.\n");
		return;
	}

	int max_shmem_optin = 0;
	if (cudaDeviceGetAttribute(&max_shmem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id) != cudaSuccess)
	{
		fprintf(stderr, "Legacy RK bounds: cannot query opt-in shared memory.\n");
		return;
	}

	const int lane_limit = block_size / 2;
	const int single_block_limit = block_size;
	const int fixed_buffer_limit = LEGACY_RK_FIXED_MAX_N;
	const int static_shmem_feasible = (LEGACY_RK_STATIC_SHMEM_BYTES <= (size_t)max_shmem_optin) ? 1 : 0;
	int effective_limit = lane_limit;
	if (single_block_limit < effective_limit)
	{
		effective_limit = single_block_limit;
	}
	if (fixed_buffer_limit < effective_limit)
	{
		effective_limit = fixed_buffer_limit;
	}

	printf("Legacy RK bounds summary: lane_limit=%d, single_block_limit=%d, fixed_buffer_limit=%d, static_shmem_bytes=%zu, static_shmem_feasible=%d, effective_limit=%d\n",
		lane_limit,
		single_block_limit,
		fixed_buffer_limit,
		(size_t)LEGACY_RK_STATIC_SHMEM_BYTES,
		static_shmem_feasible,
		effective_limit);
}

void set_output_dir(evo_parameters& evo_params) {
	int Nmax = 0;
	std::string dirstem = std::string("./output_");
	while (true) {
		std::string candidate_dir = dirstem + std::to_string(Nmax);
		if (std::filesystem::exists(candidate_dir)) {
			Nmax++;
		} else {
			break;
		}
	}

	const std::string out_dir = dirstem + std::to_string(Nmax);
	int written = snprintf(evo_params.data_out, sizeof(evo_params.data_out), "%s", out_dir.c_str());
	if (written < 0 || (size_t)written >= sizeof(evo_params.data_out)) {
		fprintf(stderr, "Error: output directory name too long: %s\n", out_dir.c_str());
		exit(1);
	}
}

static std::string normalize_relative_dir(const std::string& dir)
{
    if (dir.empty())
    {
        return std::string("./");
    }

    if (dir.rfind("./", 0) == 0 || dir[0] == '/')
    {
        return dir;
    }

    return std::string("./") + dir;
}

static void ensure_directory_exists(const std::string& dir)
{
    std::string path = normalize_relative_dir(dir);
    struct stat st;
    if (stat(path.c_str(), &st) == 0)
    {
        if (!S_ISDIR(st.st_mode))
        {
            fprintf(stderr, "Error: path exists and is not a directory: %s\n", path.c_str());
            exit(1);
        }
        return;
    }

    if (mkdir(path.c_str(), 0755) != 0 && errno != EEXIST)
    {
        fprintf(stderr, "Error: cannot create directory %s\n", path.c_str());
        exit(1);
    }
}

void read_simulation_parameters(const char* filename, ID_parameters& id_params, evo_parameters& evo_params)
{
	id_params = ID_parameters{};
	evo_params = evo_parameters{};
	id_params.data_read = 0;
	strncpy(evo_params.data_in, "input", sizeof(evo_params.data_in) - 1);
	strncpy(evo_params.data_out, "output", sizeof(evo_params.data_out) - 1);

	FILE* f = fopen(filename, "r");
	if (!f) {
		fprintf(stderr, "Error: cannot open parameter file %s\n", filename);
		exit(1);
	}

	char line[256];
	while (fgets(line, sizeof(line), f)) {
		trim(line);
		if (line[0] == '\0' || line[0] == '#') {
			continue;
		}

		char key[128], eq[4], val[128];
		if (sscanf(line, "%127s %3s %127s", key, eq, val) != 3) {
			continue;
		}
		if (strcmp(eq, "=") != 0) {
			continue;
		}

		if (strcmp(key, "data_in") == 0) {
			strncpy(evo_params.data_in, val, sizeof(evo_params.data_in) - 1);
			evo_params.data_in[sizeof(evo_params.data_in) - 1] = '\0';
		} else if (strcmp(key, "data_out") == 0) {
			strncpy(evo_params.data_out, val, sizeof(evo_params.data_out) - 1);
			evo_params.data_out[sizeof(evo_params.data_out) - 1] = '\0';
		} else if (strcmp(key, "N") == 0) {
			id_params.n_modes = atoi(val);
		} else if (strcmp(key, "mu0") == 0) {
			id_params.mu0 = atof(val);
		} else if (strcmp(key, "mu1") == 0) {
			id_params.mu1 = atof(val);
		} else if (strcmp(key, "mu2") == 0) {
			id_params.mu2 = atof(val);
		} else if (strcmp(key, "beta0") == 0) {
			id_params.beta0 = atof(val);
		} else if (strcmp(key, "beta1") == 0) {
			id_params.beta1 = atof(val);
		} else if (strcmp(key, "beta2") == 0) {
			id_params.beta2 = atof(val);
		} else if (strcmp(key, "eta") == 0) {
			id_params.eta = atof(val);
		} else if (strcmp(key, "rho") == 0) {
			id_params.rho = atof(val);
		} else if (strcmp(key, "data_read") == 0) {
			id_params.data_read = atoi(val);			
		} else if (strcmp(key, "ID_type") == 0) {
			id_params.ID_type = atoi(val);
		} else if (strcmp(key, "Nt") == 0) {
			evo_params.Nt = atoi(val);
		} else if (strcmp(key, "Nsave") == 0) {
			evo_params.Nsave = atoi(val);
		} else if (strcmp(key, "N_total") == 0) {
			evo_params.N_total = atoi(val);
		} else if (strcmp(key, "dt") == 0) {
			evo_params.dt = atof(val);
		} else if (strcmp(key, "theta") == 0) {
			evo_params.theta = atof(val);
		} else if (strcmp(key, "sigma") == 0) {
			evo_params.sigma = atof(val);
		} else if (strcmp(key, "rng_seed") == 0) {
			evo_params.rng_seed = strtoull(val, nullptr, 10);
		}
	}

	fclose(f);

	ensure_directory_exists(evo_params.data_in);
	//ensure_directory_exists(evo_params.data_out);
	set_output_dir(evo_params);
}

void write_simulation_parameters(const char* filename, const ID_parameters& id_params, const evo_parameters& evo_params)
{

	ensure_directory_exists(evo_params.data_out);

	FILE* f = fopen(filename, "w");
	if (!f) {
		fprintf(stderr, "Error: cannot open parameter file for writing: %s\n", filename);
		exit(1);
	}
	fprintf(f, "###############################\n");
	fprintf(f, "#### Simulation parameters ####\n");
	fprintf(f, "###############################\n");
	fprintf(f, "\n");
	fprintf(f, " ### Initial data parameters ###\n");
	fprintf(f, "N = %d\n", id_params.n_modes);
	fprintf(f, "mu0 = %g\n", id_params.mu0);
	fprintf(f, "mu1 = %g\n", id_params.mu1);
	fprintf(f, "mu2 = %g\n", id_params.mu2);
	fprintf(f, "beta0 = %g\n", id_params.beta0);
	fprintf(f, "beta1 = %g\n", id_params.beta1);
	fprintf(f, "beta2 = %g\n", id_params.beta2);
	fprintf(f, "eta = %g\n", id_params.eta);
	fprintf(f, "rho = %g\n", id_params.rho);
	fprintf(f, "data_read = %d\n", id_params.data_read);
	fprintf(f, "ID_type = %d\n", id_params.ID_type);
	fprintf(f, "\n");
	fprintf(f, " ### Evolution parameters ###\n");
	fprintf(f, "Nt = %d\n", evo_params.Nt);
	fprintf(f, "Nsave = %d\n", evo_params.Nsave);
	fprintf(f, "N_total = %d\n", evo_params.N_total);
	fprintf(f, "dt = %g\n", evo_params.dt);
	fprintf(f, "\n");
	fprintf(f, "### Stochastic model parameters ###\n");
	fprintf(f, "theta = %g\n", evo_params.theta);
	fprintf(f, "sigma = %g\n", evo_params.sigma);
	fprintf(f, "rng_seed = %llu\n", evo_params.rng_seed);

	if (fclose(f) != 0) {
		fprintf(stderr, "Error: cannot close parameter file: %s\n", filename);
		exit(1);
	}
}
