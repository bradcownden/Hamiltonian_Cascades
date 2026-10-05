// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Anxo Biasi, Brad Cownden, Oleg Evnin, Alvaro Iturbe Jabaloyes
//
// Single-block RK4 time step (default path, N <= 512).
// See README.md and LICENSE for details.

#include "main.cuh"

__global__ __launch_bounds__(1024, 1) void RK_step(double* A, double* B, double* C, double dt, double* out_A, double* out_B, int N, int Nt)
{
	__shared__ double sA[LEGACY_RK_FIXED_MAX_N];
	__shared__ double sB[LEGACY_RK_FIXED_MAX_N];
	__shared__ double sA_0[LEGACY_RK_FIXED_MAX_N];
	__shared__ double sB_0[LEGACY_RK_FIXED_MAX_N];
	__shared__ double sFr_1[LEGACY_RK_FIXED_MAX_N];
	__shared__ double sFi_1[LEGACY_RK_FIXED_MAX_N];
	__shared__ double sFr_2[LEGACY_RK_FIXED_MAX_N];
	__shared__ double sFi_2[LEGACY_RK_FIXED_MAX_N];
	__shared__ double sFr_3[LEGACY_RK_FIXED_MAX_N];
	__shared__ double sFi_3[LEGACY_RK_FIXED_MAX_N];

	double dA, dB, AK, BK, AI, AJ, BI, BJ;
	int K, I, J, loop, kt;
	const int tid = threadIdx.x;
	int L = tid;
	const int g_id = tid + blockIdx.x * blockDim.x;
	int FF = 0, II = 0, split = 0;
	int odd = 0, split_odd = 0, split_odd_2 = 0;
	int control = blockDim.x / N;
	int control_L = tid / N;
	if (N > LEGACY_RK_FIXED_MAX_N)
	{
		return;
	}
	if (control < 1)
	{
		control = 1;
	}
	if (control > 2)
	{
		control = 2;
	}
	const bool active = (control_L < control);
	double Coef;

	//------------------------- Open: load to shared memory -----------------
	if (g_id < N)
	{
		sA[tid] = A[tid];
		sB[tid] = B[tid];

		sA_0[tid] = sA[tid];
		sB_0[tid] = sB[tid];

	}
	//------------------------- End: load to shared memory -----------------

	__syncthreads();

	if (active)
	{
		L = L - control_L * N;

		split = N / control;

		II = split * control_L;
		FF = split * (control_L + 1);

		if (control_L == (control - 1))
		{
			FF = N;
		}


		//----- Open: L parity -----------
		if (L % 2 == 0)
		{
			odd = 0; //If L is even
		}
		else
		{
			odd = 1; //If L is odd
		}

		if (II % 2 == 0)
		{
			split_odd = 0; //If L is even
		}
		else
		{
			split_odd = 1; //If L is odd
		}

		if (FF % 2 == 0)
		{
			split_odd_2 = 0; //If L is even
		}
		else
		{
			split_odd_2 = 1; //If L is odd
		}
		//----- End: L parity ---------------

	}


	if (active)
	{

		// ------------ Loop of Nt steps performed inside the GPU ---------------
		for (loop = 0; loop < Nt; loop++)
		{
			// ------------ Each iteration is a substep of the Runge-Kutta --------
			for (kt = 0; kt < 4; kt++)
			{

				dA = 0.;
				dB = 0.;

				// ------------- Iterations over K index --------
				for (K = II; K < FF; K++)
				{
					AK = sA[K];  // We save A[k] value in a scalar because it will be used many times
					BK = sB[K];

					// --------- Iterations over I index ---------
					for (I = ((L + K) / 2 + 1); ((I < N) && (I < (L + K + 1))); I++)
					{
						J = L + K - I;

						//IMPORTANT: The best two options are C[ (K*N*N + I*N + L) ] and C[ (I*N*N + K*N + L) ], We don't know which of these two is the best.
						Coef = C[(K * N * N + I * N + L)];

						AI = sA[I];
						AJ = sA[J];
						BI = sB[I];
						BJ = sB[J];


						dA += -2. * Coef * (-AI * BJ * AK + AI * AJ * BK - BI * AJ * AK - BI * BJ * BK);
						dB += -2. * Coef * (-BI * BJ * AK + BI * AJ * BK + AI * BJ * BK + AI * AJ * AK);

					} //End for I

				} //End for K


				// ---------- Iterations over K index when (L + K) = 2n ---------


				for (K = (odd + II + split_odd); K < (FF + split_odd_2); K += 2)
				{
					I = (L + K) / 2;

					//IMPORTANT: The best two options are C[ (K*N*N + I*N + L) ] and C[ (I*N*N + K*N + L) ], We don't know which of these two is the best.
					Coef = C[(K * N * N + I * N + L)];

					AI = sA[I];
					AK = sA[K];
					BI = sB[I];
					BK = sB[K];

					dA += -1. * Coef * (-2. * AI * BI * AK + AI * AI * BK - BI * BI * BK);
					dB += -1. * Coef * (-BI * BI * AK + 2. * BI * AI * BK + AI * AI * AK);

				} //End for K

				//--------- This synchronization is needed to guarantee that sA[L] values are not modifficated before all threads have finished working


				__syncthreads();

				if (L < N && control_L == 0)
				{
					if (kt == 0)
					{
						sFr_1[L] = dt * dA; // Update coefficients values
						sFi_1[L] = dt * dB;

						sA[L] += RK_coef[0] * dt * dA;
						sB[L] += RK_coef[0] * dt * dB;
					}

					if (kt == 1)
					{
						sFr_2[L] = dt * dA; // Update coefficients values
						sFi_2[L] = dt * dB;

						sA[L] = sA_0[L] + RK_coef[1] * dt * dA;
						sB[L] = sB_0[L] + RK_coef[1] * dt * dB;
					}

					if (kt == 2)
					{
						sFr_3[L] = dt * dA; // Update coefficients values
						sFi_3[L] = dt * dB;

						sA[L] = sA_0[L] + RK_coef[2] * dt * dA;
						sB[L] = sB_0[L] + RK_coef[2] * dt * dB;
					}

					if (kt == 3)
					{
						sA[L] = sA_0[L] + 1. / 6. * (sFr_1[L] + 2. * sFr_2[L] + 2. * sFr_3[L] + dt * dA);
						sB[L] = sB_0[L] + 1. / 6. * (sFi_1[L] + 2. * sFi_2[L] + 2. * sFi_3[L] + dt * dB);

					}
				}

				__syncthreads();


				if (L < N && control_L == 1)
				{
					if (kt == 0)
					{
						sFr_1[L] += dt * dA; // Update coefficients values
						sFi_1[L] += dt * dB;

						sA[L] += RK_coef[0] * dt * dA;
						sB[L] += RK_coef[0] * dt * dB;
					}

					if (kt == 1)
					{
						sFr_2[L] += dt * dA; // Update coefficients values
						sFi_2[L] += dt * dB;

						sA[L] += RK_coef[1] * dt * dA;
						sB[L] += RK_coef[1] * dt * dB;
					}

					if (kt == 2)
					{
						sFr_3[L] += dt * dA; // Update coefficients values
						sFi_3[L] += dt * dB;

						sA[L] += RK_coef[2] * dt * dA;
						sB[L] += RK_coef[2] * dt * dB;
					}

					if (kt == 3)
					{
						sA[L] += 1. / 6. * (dt * dA);
						sB[L] += 1. / 6. * (dt * dB);

						sA_0[L] = sA[L];
						sB_0[L] = sB[L];
					}
				}

				__syncthreads(); // New needed synchronization

			} //End for kt
		} //End for loop
	} // End active/lane compute

	if (control_L == 0 && L < N)
	{
		out_A[L] = sA[L];
		out_B[L] = sB[L];
	}

}
