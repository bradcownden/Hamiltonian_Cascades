// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Anxo Biasi, Brad Cownden, Oleg Evnin, Alvaro Iturbe Jabaloyes
//
// CUDA kernels for the interaction coefficients: deterministic part,
// stochastic (Ornstein-Uhlenbeck) part, and odd/even (n+m) corrections.
//
// Box-Muller Gaussian sampling follows GPU Gems 3, ch. 37
// (https://developer.nvidia.com/gpugems/gpugems3).
// See README.md and LICENSE for details.

#include "main.cuh"


// Kronecker delta helper used by both host/device math.
__host__ __device__ double KroneckerDelta(int a, int b)
{
	return (a == b) ? 1.0 : 0.0;
}

// Deterministic interaction coefficients.
__global__ void build_deterministic_coefficients(const double* logf, double* C, int N, double mu0, double mu1, double mu2)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = N * N * N;
	if (idx >= total)
	{
		return;
	}

	int n = idx / (N * N);
	int rem = idx % (N * N);
	int m = rem / N;
	int k = rem % N;

	if (m > n)
	{
		return;
	}
	if (k < m || k > (n + m) / 2)
	{
		return;
	}

	int l = n + m - k;
	if (l < 0 || l >= N)
	{
		return;
	}

	if (n == 0 || m == 0 || k == 0 || l == 0)
	{
		double log_ratio = logf[n] + logf[m] + logf[k] + logf[l] - 2.0 * logf[n + m];
		double coef = (mu2 * (n * m + k * l) + mu1 * (n + m) + mu0) * exp(log_ratio);

		//coef = 1.0;

		C[m * N * N + k * N + n] = coef;
		C[n * N * N + k * N + m] = coef;
		C[m * N * N + l * N + n] = coef;
		C[n * N * N + l * N + m] = coef;
		C[l * N * N + n * N + k] = coef;
		C[l * N * N + m * N + k] = coef;
		C[k * N * N + n * N + l] = coef;
		C[k * N * N + m * N + l] = coef;
	}
}

// GPU-based pseudo-random coefficient generator for the stochastic part, based on the (n, m, k) indices and a step seed. The distribution is uniform in [0, 1].
__device__ double uniform_random_from_index(int n, int m, int k, int step_seed)
{
	unsigned int x = (unsigned int)(n * 73856093u) ^ (unsigned int)(m * 19349663u) ^ (unsigned int)(k * 83492791u) ^ (unsigned int)(step_seed * 2654435761u);
	x ^= x >> 13;
	x *= 1274126177u;
	x ^= x >> 16;
	double u = (double)(x & 0x00FFFFFFu) / 16777215.0;
	return u;
}

// GPU-based pseudo-random coefficient generator for the stochastic part in [-1, 1].
__device__ double random_coef_from_index(int n, int m, int k, int step_seed)
{
	unsigned int x = (unsigned int)(n * 73856093u) ^ (unsigned int)(m * 19349663u) ^ (unsigned int)(k * 83492791u) ^ (unsigned int)(step_seed * 2654435761u);
	x ^= x >> 13;
	x *= 1274126177u;
	x ^= x >> 16;
	double u = (double)(x & 0x00FFFFFFu) / 16777215.0;
	return 2.0 * u - 1.0;
}

// GPU-based pseudo-random Gaussian generator using the Box-Muller transform,
// based on the (n, m, k) indices and per-call clock entropy.
// https://developer.nvidia.com/gpugems/gpugems3
__device__ double gaussian_random_from_index(int n, int m, int k)
{
	unsigned int base_seed = (unsigned int)(clock64() & 0xffffffffu);
	double u1 = uniform_random_from_index(n, m, k, (int)base_seed);
	double u2 = uniform_random_from_index(k, n, m, (int)(base_seed ^ 0x9e3779b9u));
	if (u1 <= 0.0)
	{
		u1 = 1e-12;
	}

	double r = sqrt(-2.0 * log(u1));
	double theta = 2.0 * M_PI * u2;

	return r * cos(theta);
}

// Random interaction coefficients for the stochastic part,
// updated every RK step.
__global__ void build_random_coefficients(double* C, int step_seed, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = N * N * N;
	if (idx >= total)
	{
		return;
	}

	int n = idx / (N * N);
	int rem = idx % (N * N);
	int m = rem / N;
	int k = rem % N;

	if (n < 1 || m < 1 || m > n)
	{
		return;
	}
	if (k < m || k > (n + m) / 2)
	{
		return;
	}

	int l = n + m - k;
	if (l < 0 || l >= N)
	{
		return;
	}

	double coef = random_coef_from_index(n, m, k, step_seed);

	C[m * N * N + k * N + n] = coef;
	C[n * N * N + k * N + m] = coef;
	C[m * N * N + l * N + n] = coef;
	C[n * N * N + l * N + m] = coef;
	C[l * N * N + n * N + k] = coef;
	C[l * N * N + m * N + k] = coef;
	C[k * N * N + n * N + l] = coef;
	C[k * N * N + m * N + l] = coef;
}

__global__ void stochastic_coefficients(double* C, int step_seed, double theta, double sigma, double dt, int N)
{
	int n = blockIdx.y * blockDim.y + threadIdx.y+1; // 1..N-1
	int m = blockIdx.x * blockDim.x + threadIdx.x+1; // 1..N-1

	if (n >= N) return;
	if (m > n) return; // 1 <= m <= n

	int k_min = m;
  int k_max = (n + m) / 2;
  if (k_min >= N) return;


	// == Stochastic profile ==
	// Stochastic profile: C(t+dt) = C(t) - THETA dt C(t) + SIGMA sqrt(2 THETA dt) * gaussian_random

	double decay = 1.0 - theta * dt;
	double noise_scale = sigma * sqrt(2.0 * theta * dt);

	for (int k = k_min; k <= k_max && k < N; k++)
	{
		int l = n + m - k;
		if (l < 0 || l >= N) continue;

		double coef = gaussian_random_from_index(n, m, k);
		double old = C[m*N*N + k*N + n];
		double noise = noise_scale * coef;
		double newval = old * decay + noise;

		//newval = 1.0;
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

// Initialize stochastic coefficients with Gaussian random values before the first RK step, to avoid starting with zero stochastic coefficients.
__global__ void initialize_stochastic_coefficients(double* C, int N)
{
	int n = blockIdx.y * blockDim.y + threadIdx.y+1; // 1..N-1
	int m = blockIdx.x * blockDim.x + threadIdx.x+1; // 1..N-1

	if (n >= N) return;
	if (m > n) return; // 1 <= m <= n

	int k_min = m;
  int k_max = (n + m) / 2;
  if (k_min >= N) return;

	// Initialize with Gaussian random values with mean 0 and stddev 1
	for (int k = k_min; k <= k_max && k < N; k++)
	{
		int l = n + m - k;
		if (l < 0 || l >= N) continue;

		double coef = gaussian_random_from_index(n, m, k);

		C[m*N*N + k*N + n] = coef;
		C[n*N*N + k*N + m] = coef;
		C[m*N*N + l*N + n] = coef;
		C[n*N*N + l*N + m] = coef;
		C[l*N*N + n*N + k] = coef;
		C[l*N*N + m*N + k] = coef;
		C[k*N*N + n*N + l] = coef;
		C[k*N*N + m*N + l] = coef;
	}

}

// Odd (n+m) correction coefficients.
// Parallelization strategy: one thread per (M, m) pair with M odd.
// Each raw term f[k]*f[M-k]/(f[n]*f[m]) is folded into a single exp() of a
// log-difference so no intermediate f[.] value ever needs to leave double
// range, even when k or M-k is large.
__global__ void apply_odd_corrections(const double* logf, double* C, int N, double b0, double b1, double b2)
{
	int m = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int q = blockIdx.y * blockDim.y + threadIdx.y;

	if (q >= (N - 1) || m >= N)
	{
		return;
	}

	int M = 2 * q + 3;
	if (m > M / 2)
	{
		return;
	}

	int n = M - m;
	if (!(n < N && m < N))
	{
		return;
	}

	double log_nm = logf[n] + logf[m];

	double sum1 = 0.0;
	for (int k = 1; k < m; k++)
	{
		double ratio = exp(logf[k] + logf[M - k] - log_nm);
		sum1 += ratio * C[m * N * N + k * N + n];
	}

	double sum2 = 0.0;
	for (int k = m + 1; k <= (M - 1) / 2; k++)
	{
		double ratio = exp(logf[k] + logf[M - k] - log_nm);
		sum2 += ratio * C[m * N * N + k * N + n];
	}

	double coef = 0.5 * (b0 + b1 * (n + m) + b2 * n * m)
		- (sum1 + (1.0 - KroneckerDelta(m, (n + m - 1) / 2)) * sum2);

	C[m * N * N + n * N + n] = coef;
	C[n * N * N + n * N + m] = coef;
	C[m * N * N + m * N + n] = coef;
	C[n * N * N + m * N + m] = coef;
}

// Even (n+m) correction coefficients.
// Parallelization strategy: one thread per (M, m) pair with M even.
// See apply_odd_corrections for why terms are folded through exp(log-diff).
__global__ void apply_even_corrections(const double* logf, double* C, int N, double b0, double b1, double b2)
{
	int m = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int q = blockIdx.y * blockDim.y + threadIdx.y;

	if (q >= (N - 1) || m >= N)
	{
		return;
	}

	int M = 2 * q + 2;
	if (m > M / 2)
	{
		return;
	}

	int n = M - m;
	if (!(n < N && m < N))
	{
		return;
	}

	double log_nm = logf[n] + logf[m];

	double sum1 = 0.0;
	for (int k = 1; k < m; k++)
	{
		double ratio = exp(logf[k] + logf[M - k] - log_nm);
		sum1 += ratio * C[m * N * N + k * N + n];
	}

	double sum2 = 0.0;
	for (int k = m + 1; k <= M / 2 - 1; k++)
	{
		double ratio = exp(logf[k] + logf[M - k] - log_nm);
		sum2 += ratio * C[m * N * N + k * N + n];
	}

	double delta_m_half = KroneckerDelta(m, M / 2);
	double delta_nm_half_minus_1 = KroneckerDelta(m, (n + m) / 2 - 1);

	double half_term = 0.0;
	if (delta_m_half == 0.0)
	{
		double ratio_half = exp(2.0 * logf[M / 2] - log_nm);
		half_term = ratio_half * C[m * N * N + (M / 2) * N + n];
	}

	double coef = ((b0 + b1 * (n + m) + b2 * n * m)
		- (2.0 * sum1 + 2.0 * (1.0 - delta_nm_half_minus_1) * sum2 + (1.0 - delta_m_half) * half_term))
		/ (2.0 - delta_m_half);

	C[m * N * N + n * N + n] = coef;
	C[n * N * N + n * N + m] = coef;
	C[m * N * N + m * N + n] = coef;
	C[n * N * N + m * N + m] = coef;
}

__global__ void init_first_non_finite(int* first_idx)
{
	if (blockIdx.x == 0 && threadIdx.x == 0)
	{
		*first_idx = -1;
	}
}

__global__ void find_first_non_finite(const double* data, int count, int* first_idx)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= count)
	{
		return;
	}

	double v = data[idx];
	if (!isfinite(v))
	{
		atomicCAS(first_idx, -1, idx);
	}
}
