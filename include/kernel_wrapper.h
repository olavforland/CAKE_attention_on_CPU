// #pragma once


#include "common.h"

// kernel helper functions
inline void cake_sgemm_ukernel(float* A_p, float* B_p, float* C_p, 
	int m_r, int n_r, int k_c_t, cake_cntx_t* cake_cntx) {


cake_sgemm_armv8_8x12(A_p, B_p, C_p, m_r, n_r, k_c_t);

}

inline void cake_sgemm_small_ukernel(float* A_p, float* B_p, float* C_p, 
	int m_r, int n_r, int k_c_t, int M, int K, int N) {

}


