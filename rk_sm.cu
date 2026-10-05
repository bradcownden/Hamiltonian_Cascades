// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Anxo Biasi, Brad Cownden, Oleg Evnin, Alvaro Iturbe Jabaloyes
//
// Seeded stochastic-coefficient kernels and the multi-block RK4 time step
// (enabled with CASCADES_RK_SM=1).
//
// Random numbers use cuRAND's Philox4x32-10 counter-based generator
// (J. K. Salmon et al., "Parallel random numbers: as easy as 1, 2, 3", SC '11).
// See README.md and LICENSE for details.

// Phase 0 of the RK restructuring effort (see repo memory notes).
//
// The existing stochastic coefficient kernels (kernels.cu) seed their
// Gaussian noise from clock64(), so the physical forcing is wall-clock
// dependent and cannot be reproduced across runs -- not even against
// another run of the same binary. That is fine for "the physics must be
// non-deterministic" in production, but it also means there is no way to
// regression-test any future RK_step rewrite against a known-good
// trajectory.
//
// This file adds a seeded alternative: a 64-bit run_seed, combined with
// (step_seed, n, m, k), drives cuRAND's Philox4x32-10 counter-based
// generator instead of clock64(). By default the run_seed is drawn from
// real entropy at startup (so production remains physically
// non-deterministic, same as today), but it can be pinned via
// parameters.txt (rng_seed = <nonzero>) to get a fully reproducible run
// for testing. kernels.cu and rk.cu are left untouched.

#include "main.cuh"
#include <curand_kernel.h>
#include <random>
#include <chrono>
#include <stdlib.h>

unsigned long long resolve_run_seed(unsigned long long configured_seed)
{
	if (configured_seed != 0ULL)
	{
		return configured_seed;
	}

	std::random_device rd;
	unsigned long long seed = ((unsigned long long)rd() << 32) ^ (unsigned long long)rd();
	seed ^= (unsigned long long)std::chrono::high_resolution_clock::now().time_since_epoch().count();
	if (seed == 0ULL)
	{
		// Astronomically unlikely, but never hand back the "auto" sentinel.
		seed = 0x9E3779B97F4A7C15ULL;
	}
	return seed;
}

// Deterministic replacement for gaussian_random_from_index()'s clock64()
// seeding: (run_seed, step_seed, n, m, k) select a unique Philox
// counter/subsequence, so the draw depends only on its arguments, not on
// wall-clock/scheduling.
__device__ double gaussian_random_from_seed(unsigned long long run_seed, int step_seed, int n, int m, int k)
{
	unsigned long long subsequence = ((unsigned long long)(unsigned int)n * 0x9E3779B97F4A7C15ULL)
		^ ((unsigned long long)(unsigned int)m * 0xC2B2AE3D27D4EB4FULL)
		^ ((unsigned long long)(unsigned int)k * 0x165667B19E3779F9ULL);
	unsigned long long offset = (unsigned long long)(unsigned int)step_seed;

	curandStatePhilox4_32_10_t state;
	curand_init(run_seed, subsequence, offset, &state);
	return curand_normal_double(&state);
}

// Seeded counterpart of initialize_stochastic_coefficients_v2 (kernels.cu).
// Tag the draw with a step_seed of -1 so it can never collide with an
// actual evolution step (step_seed = i*Nsave + j >= 0).
__global__ void initialize_stochastic_coefficients_v3(const double* logf, double* C, unsigned long long run_seed, int N, double mu0, double mu1, double mu2)
{
	int n = blockIdx.y * blockDim.y + threadIdx.y + 1; // 1..N-1
	int m = blockIdx.x * blockDim.x + threadIdx.x + 1; // 1..N-1

	if (n >= N) return;
	if (m > n) return; // 1 <= m <= n

	int k_min = m;
	int k_max = (n + m) / 2;
	if (k_min >= N) return;

	double known = mu2 * n * m + mu1 * (n + m) + mu0;

	for (int k = k_min; k <= k_max && k < N; k++)
	{
		int l = n + m - k;
		if (l < 0 || l >= N) continue;

		double log_ratio = logf[n] + logf[m] + logf[k] + logf[l] - 2.0 * logf[n + m];
		double coef = known * exp(0.5 * log_ratio);

		double random_coef = gaussian_random_from_seed(run_seed, -1, n, m, k);

		C[m*N*N + k*N + n] = coef * random_coef;
		C[n*N*N + k*N + m] = coef * random_coef;
		C[m*N*N + l*N + n] = coef * random_coef;
		C[n*N*N + l*N + m] = coef * random_coef;
		C[l*N*N + n*N + k] = coef * random_coef;
		C[l*N*N + m*N + k] = coef * random_coef;
		C[k*N*N + n*N + l] = coef * random_coef;
		C[k*N*N + m*N + l] = coef * random_coef;
	}
}

// Seeded counterpart of stochastic_coefficients_v2 (kernels.cu): identical
// math, but the noise is a deterministic function of (run_seed, step_seed,
// n, m, k) instead of clock64().
__global__ void stochastic_coefficients_v3(const double* logf, double* C, unsigned long long run_seed, int step_seed, double theta, double sigma, double dt, int N, double mu0, double mu1, double mu2)
{
	int n = blockIdx.y * blockDim.y + threadIdx.y + 1; // 1..N-1
	int m = blockIdx.x * blockDim.x + threadIdx.x + 1; // 1..N-1

	if (n >= N) return;
	if (m > n) return; // 1 <= m <= n

	int k_min = m;
	int k_max = (n + m) / 2;
	if (k_min >= N) return;

	double known = mu2 * n * m + mu1 * (n + m) + mu0;

	double decay = 1.0 - theta * dt;
	double noise_scale = sigma * sqrt(2.0 * theta * dt);

	for (int k = k_min; k <= k_max && k < N; k++)
	{
		int l = n + m - k;
		if (l < 0 || l >= N) continue;

		double log_ratio = logf[n] + logf[m] + logf[k] + logf[l] - 2.0 * logf[n + m];
		double F = known * exp(0.5 * log_ratio);

		double random_coef = gaussian_random_from_seed(run_seed, step_seed, n, m, k);
		double noise = noise_scale * random_coef;
		double newval = decay + noise;

		C[m*N*N + k*N + n] = F * newval;
		C[n*N*N + k*N + m] = F * newval;
		C[m*N*N + l*N + n] = F * newval;
		C[n*N*N + l*N + m] = F * newval;
		C[l*N*N + n*N + k] = F * newval;
		C[l*N*N + m*N + k] = F * newval;
		C[k*N*N + n*N + l] = F * newval;
		C[k*N*N + m*N + l] = F * newval;
	}
}

// Seeded Ornstein-Uhlenbeck update of the stochastic coefficients (the
// memory-carrying counterpart of the legacy stochastic_coefficients kernel
// in kernels.cu; v2/v3 are deliberately memoryless). One Euler-Maruyama
// step of
//   dC = -theta C dt + sigma sqrt(2 theta) dW
// i.e. C(t+dt) = (1 - theta dt) C(t) + sigma sqrt(2 theta dt) xi,
// with xi ~ N(0,1) drawn deterministically from (run_seed, step_seed, n, m, k).
// Stationary distribution: N(0, sigma^2). No mode-dependent envelope.
// Each quartet {n,m} <-> {k,l=n+m-k} is owned by exactly one thread
// (m = global min, k in [m, (n+m)/2]), so the read of the old value and
// the 8 symmetric writes are race-free.
__global__ void stochastic_coefficients_sm(double* C, unsigned long long run_seed, int step_seed, double theta, double sigma, double dt, int N)
{
	int n = blockIdx.y * blockDim.y + threadIdx.y + 1; // 1..N-1
	int m = blockIdx.x * blockDim.x + threadIdx.x + 1; // 1..N-1

	if (n >= N) return;
	if (m > n) return; // 1 <= m <= n

	int k_min = m;
	int k_max = (n + m) / 2;
	if (k_min >= N) return;


	double decay = 1.0 - theta * dt;
	double noise_scale = sigma * sqrt(2.0 * theta * dt);

	for (int k = k_min; k <= k_max && k < N; k++)
	{
		int l = n + m - k;
		if (l < 0 || l >= N) continue;

		double old = C[m*N*N + k*N + n];
		double random_coef = gaussian_random_from_seed(run_seed, step_seed, n, m, k);
		double noise = noise_scale * random_coef;
		double newval = decay * old + noise;

		C[m*N*N + k*N + n] = newval;
		C[n*N*N + k*N + m] = newval;
		C[m*N*N + l*N + n] = newval;
		C[n*N*N + l*N + m] = newval;
		C[l*N*N + n*N + k] = newval;
		C[l*N*N + m*N + k] = newval;
		C[k*N*N + n*N + l] = newval;
		C[k*N*N + m*N + l] = newval;
	}
}

// Initial condition for stochastic_coefficients_sm: draw every quartet from
// the OU stationary distribution N(0, sigma^2). Tagged with step_seed = -1 so
// the draw can never coincide with an evolution step (step_seed >= 0).
__global__ void initialize_stochastic_coefficients_sm(double* C, unsigned long long run_seed, double sigma, int N)
{
	int n = blockIdx.y * blockDim.y + threadIdx.y + 1; // 1..N-1
	int m = blockIdx.x * blockDim.x + threadIdx.x + 1; // 1..N-1

	if (n >= N) return;
	if (m > n) return; // 1 <= m <= n

	int k_min = m;
	int k_max = (n + m) / 2;
	if (k_min >= N) return; 

	for (int k = k_min; k <= k_max && k < N; k++)
	{
		int l = n + m - k;
		if (l < 0 || l >= N) continue;

		double random_coef = sigma * gaussian_random_from_seed(run_seed, -1, n, m, k);

		C[m*N*N + k*N + n] = random_coef;
		C[n*N*N + k*N + m] = random_coef;
		C[m*N*N + l*N + n] = random_coef;
		C[n*N*N + l*N + m] = random_coef;
		C[l*N*N + n*N + k] = random_coef;
		C[l*N*N + m*N + k] = random_coef;
		C[k*N*N + n*N + l] = random_coef;
		C[k*N*N + m*N + l] = random_coef;
	}
}

// ============================================================================
// Multi-block RK4 (CASCADES_RK_SM=1)
//
// The right-hand side for mode L is a double sum over K and I of
// C[K,I,L] * g(A,B; I, J=L+K-I, K). The previous version of this file used
// one thread per mode and looped over the whole K range, so a run with
// N=400 and 256 threads/block launched 2 blocks in total and left every
// other SM idle (the legacy RK_step launches exactly 1 block).
//
// Here the K range is split into k_splits chunks and each (mode block,
// K chunk) pair becomes one block of the 2-D grid, so the grid scales with
// N^2 instead of N. Each block writes its partial sum for every mode into
// part[blockIdx.y][L]; rk_sm_combine then sums the partials in fixed
// chunk order, so results are bit-reproducible for a given (device, plan),
// which the pinned rng_seed regression runs rely on. (Using atomicAdd
// instead would make the summation order scheduler dependent.)
//
// A and B are staged in dynamic shared memory (2*N doubles); C is streamed
// from global memory with consecutive threads reading consecutive L, which
// is the coalesced direction of the C[K*N*N + I*N + L] layout.
// ============================================================================

__global__ void __launch_bounds__(RK_SM_THREADS)
rk_sm_partial(const double* __restrict__ A,
	const double* __restrict__ B,
	const double* __restrict__ C,
	double* __restrict__ partA,
	double* __restrict__ partB,
	int N,
	int k_chunk)
{
	extern __shared__ double rk_sm_shared[];
	double* sA = rk_sm_shared;
	double* sB = rk_sm_shared + N;

	for (int i = threadIdx.x; i < N; i += blockDim.x)
	{
		sA[i] = A[i];
		sB[i] = B[i];
	}
	__syncthreads();

	const int L = blockIdx.x * blockDim.x + threadIdx.x;
	if (L >= N)
	{
		return;
	}

	const int K_begin = blockIdx.y * k_chunk;
	const int K_end = min(N, K_begin + k_chunk);
	const size_t NN = (size_t)N * (size_t)N;

	// The legacy RK_step body evaluates, per (K, I, J=L+K-I) term,
	//   dA += -2 C (-AI BJ AK + AI AJ BK - BI AJ AK - BI BJ BK)
	//   dB += -2 C (-BI BJ AK + BI AJ BK + AI BJ BK + AI AJ AK)
	// With the complex product (AI + i BI)(AJ + i BJ) = P + i Q, i.e.
	// P = AI AJ - BI BJ and Q = AI BJ + BI AJ, this is exactly
	//   dA +=  2 C (AK Q - BK P),   dB += -2 C (AK P + BK Q)
	// and the diagonal I == J term is the same expression with weight 1
	// instead of 2. The factor 2 is applied once at the end. This halves
	// the fp64 instruction count, which is the bottleneck on GPUs with
	// 1/32-rate fp64 (Turing/consumer parts); results agree with the
	// legacy kernel to round-off.
	double dA = 0.0;
	double dB = 0.0;
	for (int K = K_begin; K < K_end; ++K)
	{
		const double AK = sA[K];
		const double BK = sB[K];
		const int LK = L + K;
		const int I_end = min(N, LK + 1);
		const double* Ccol = C + (size_t)K * NN + (size_t)L; // C[K, I, L] == Ccol[I * N]

		for (int I = LK / 2 + 1; I < I_end; ++I)
		{
			const int J = LK - I;
			const double Coef = __ldg(Ccol + (size_t)I * (size_t)N);
			const double AI = sA[I];
			const double AJ = sA[J];
			const double BI = sB[I];
			const double BJ = sB[J];
			const double P = AI * AJ - BI * BJ;
			const double Q = AI * BJ + BI * AJ;

			dA += Coef * (AK * Q - BK * P);
			dB -= Coef * (AK * P + BK * Q);
		}

		const int I = LK / 2;
		if (I < N && 2 * I == LK)
		{
			const double Coef = 0.5 * __ldg(Ccol + (size_t)I * (size_t)N);
			const double AI = sA[I];
			const double BI = sB[I];
			const double P = AI * AI - BI * BI;
			const double Q = 2.0 * AI * BI;

			dA += Coef * (AK * Q - BK * P);
			dB -= Coef * (AK * P + BK * Q);
		}
	}
	dA *= 2.0;
	dB *= 2.0;

	const size_t slot = (size_t)blockIdx.y * (size_t)N + (size_t)L;
	partA[slot] = dA;
	partB[slot] = dB;
}

// Sums the K-chunk partials in fixed order and applies the RK4 stage update.
// Stage 0 also snapshots the stage input into A0/B0 (the base state for the
// whole step), which replaces the former rk_sm_copy_base launch.
__global__ void rk_sm_combine(const double* __restrict__ partA,
	const double* __restrict__ partB,
	int k_splits,
	const double* __restrict__ Ain,
	const double* __restrict__ Bin,
	double* __restrict__ A0,
	double* __restrict__ B0,
	double* __restrict__ Aout,
	double* __restrict__ Bout,
	double* __restrict__ k1A,
	double* __restrict__ k1B,
	double* __restrict__ k2A,
	double* __restrict__ k2B,
	double* __restrict__ k3A,
	double* __restrict__ k3B,
	double dt,
	int N,
	int stage)
{
	const int L = blockIdx.x * blockDim.x + threadIdx.x;
	if (L >= N)
	{
		return;
	}

	double dA = 0.0;
	double dB = 0.0;
	for (int s = 0; s < k_splits; ++s)
	{
		const size_t slot = (size_t)s * (size_t)N + (size_t)L;
		dA += partA[slot];
		dB += partB[slot];
	}

	if (stage == 0)
	{
		A0[L] = Ain[L];
		B0[L] = Bin[L];
	}
	const double a0 = A0[L];
	const double b0 = B0[L];

	const double dt_dA = dt * dA;
	const double dt_dB = dt * dB;
	if (stage == 0)
	{
		k1A[L] = dt_dA;
		k1B[L] = dt_dB;
		Aout[L] = a0 + 0.5 * dt_dA;
		Bout[L] = b0 + 0.5 * dt_dB;
	}
	else if (stage == 1)
	{
		k2A[L] = dt_dA;
		k2B[L] = dt_dB;
		Aout[L] = a0 + 0.5 * dt_dA;
		Bout[L] = b0 + 0.5 * dt_dB;
	}
	else if (stage == 2)
	{
		k3A[L] = dt_dA;
		k3B[L] = dt_dB;
		Aout[L] = a0 + dt_dA;
		Bout[L] = b0 + dt_dB;
	}
	else
	{
		Aout[L] = a0 + (k1A[L] + 2.0 * k2A[L] + 2.0 * k3A[L] + dt_dA) / 6.0;
		Bout[L] = b0 + (k1B[L] + 2.0 * k2B[L] + 2.0 * k3B[L] + dt_dB) / 6.0;
	}
}

// Chooses the 2-D grid for rk_sm_partial on the current device. The K
// range is split so that the grid has roughly RK_SM_BLOCKS_PER_SM blocks
// per SM (more than that only adds partial-buffer traffic). Set
// CASCADES_RK_SM_KSPLITS=<n> to force the number of K chunks when tuning.
bool rk_sm_make_plan(int N, rk_sm_plan& plan)
{
	if (N <= 0)
	{
		fprintf(stderr, "Error: N must be positive. Current N = %d\n", N);
		return false;
	}

	int device_id = 0;
	CUDA_CHECK(cudaGetDevice(&device_id));
	int sm_count = 0;
	int max_shmem_optin = 0;
	int max_grid_y = 0;
	CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));
	CUDA_CHECK(cudaDeviceGetAttribute(&max_shmem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id));
	CUDA_CHECK(cudaDeviceGetAttribute(&max_grid_y, cudaDevAttrMaxGridDimY, device_id));

	plan.threads = RK_SM_THREADS;
	plan.l_blocks = (N + plan.threads - 1) / plan.threads;
	plan.shmem = 2 * (size_t)N * sizeof(double);

	if (plan.shmem > (size_t)max_shmem_optin)
	{
		fprintf(stderr,
			"Error: rk_sm_partial needs %zu bytes of shared memory for N=%d, device allows %d\n",
			plan.shmem, N, max_shmem_optin);
		return false;
	}
	if (plan.shmem > 48 * 1024)
	{
		CUDA_CHECK(cudaFuncSetAttribute(rk_sm_partial, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)plan.shmem));
	}

	const int target_blocks = RK_SM_BLOCKS_PER_SM * (sm_count > 0 ? sm_count : 1);
	int k_splits = (target_blocks + plan.l_blocks - 1) / plan.l_blocks;

	const char* env = getenv("CASCADES_RK_SM_KSPLITS");
	if (env && env[0] != '\0')
	{
		int forced = atoi(env);
		if (forced > 0)
		{
			k_splits = forced;
		}
	}

	if (k_splits < 1) k_splits = 1;
	if (k_splits > N) k_splits = N;
	plan.k_chunk = (N + k_splits - 1) / k_splits;
	plan.k_splits = (N + plan.k_chunk - 1) / plan.k_chunk; // drop empty trailing chunks
	if (plan.k_splits > max_grid_y)
	{
		fprintf(stderr, "Error: rk_sm K split count %d exceeds device grid limit %d\n", plan.k_splits, max_grid_y);
		return false;
	}

	printf("RK (multi-block): N=%d, SMs=%d, threads/block=%d, grid=%d x %d (mode blocks x K chunks of %d), shmem=%zu B\n",
		N, sm_count, plan.threads, plan.l_blocks, plan.k_splits, plan.k_chunk, plan.shmem);
	return true;
}

// One full RK4 step (4 stages) for Nt substeps. On return the updated state
// is in A/B; scratch is only valid within the call.
void rk_sm_step(const rk_sm_plan& plan, double* A, double* B, const double* C, rk_sm_scratch& s, double dt, int N, int Nt)
{
	const dim3 grid_partial((unsigned)plan.l_blocks, (unsigned)plan.k_splits);
	const dim3 grid_combine((unsigned)plan.l_blocks);

	for (int sub = 0; sub < Nt; ++sub)
	{
		// Stage inputs/outputs ping-pong between (A, B) and (s.tmpA, s.tmpB):
		// stage 0 reads A, writes tmp; stage 1 reads tmp, writes A; ... so
		// the final stage 3 leaves the result in A/B with no pointer swap.
		const double* in_A[4] = { A, s.tmpA, A, s.tmpA };
		const double* in_B[4] = { B, s.tmpB, B, s.tmpB };
		double* out_A[4] = { s.tmpA, A, s.tmpA, A };
		double* out_B[4] = { s.tmpB, B, s.tmpB, B };

		for (int stage = 0; stage < 4; ++stage)
		{
			rk_sm_partial<<<grid_partial, plan.threads, plan.shmem>>>(in_A[stage], in_B[stage], C, s.partA, s.partB, N, plan.k_chunk);
			CUDA_CHECK_KERNEL();
			rk_sm_combine<<<grid_combine, plan.threads>>>(s.partA, s.partB, plan.k_splits,
				in_A[stage], in_B[stage], s.A0, s.B0, out_A[stage], out_B[stage],
				s.k1A, s.k1B, s.k2A, s.k2B, s.k3A, s.k3B, dt, N, stage);
			CUDA_CHECK_KERNEL();
		}
	}
}

void rk_sm_alloc_scratch(const rk_sm_plan& plan, int N, rk_sm_scratch& s)
{
	const size_t nbytes = sizeof(double) * (size_t)N;
	CUDA_CHECK(cudaMalloc(&s.tmpA, nbytes));
	CUDA_CHECK(cudaMalloc(&s.tmpB, nbytes));
	CUDA_CHECK(cudaMalloc(&s.A0, nbytes));
	CUDA_CHECK(cudaMalloc(&s.B0, nbytes));
	CUDA_CHECK(cudaMalloc(&s.k1A, nbytes));
	CUDA_CHECK(cudaMalloc(&s.k1B, nbytes));
	CUDA_CHECK(cudaMalloc(&s.k2A, nbytes));
	CUDA_CHECK(cudaMalloc(&s.k2B, nbytes));
	CUDA_CHECK(cudaMalloc(&s.k3A, nbytes));
	CUDA_CHECK(cudaMalloc(&s.k3B, nbytes));
	CUDA_CHECK(cudaMalloc(&s.partA, nbytes * (size_t)plan.k_splits));
	CUDA_CHECK(cudaMalloc(&s.partB, nbytes * (size_t)plan.k_splits));
}

void rk_sm_free_scratch(rk_sm_scratch& s)
{
	CUDA_CHECK(cudaFree(s.tmpA));
	CUDA_CHECK(cudaFree(s.tmpB));
	CUDA_CHECK(cudaFree(s.A0));
	CUDA_CHECK(cudaFree(s.B0));
	CUDA_CHECK(cudaFree(s.k1A));
	CUDA_CHECK(cudaFree(s.k1B));
	CUDA_CHECK(cudaFree(s.k2A));
	CUDA_CHECK(cudaFree(s.k2B));
	CUDA_CHECK(cudaFree(s.k3A));
	CUDA_CHECK(cudaFree(s.k3B));
	CUDA_CHECK(cudaFree(s.partA));
	CUDA_CHECK(cudaFree(s.partB));
	s = rk_sm_scratch();
}
