#include "cake.h"


void schedule_MKN(float* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x) {

	// copy over block dims to local vars to avoid readibility ussiues with x->
	int m_r = cake_cntx->mr, n_r = cake_cntx->nr;

	int m_c = x->m_c, k_c = x->k_c, n_c = x->n_c;
	int m_c1 = x->m_c1, k_c1 = x->k_c1, n_c1 = x->n_c1;
	int k_c1_last_core = x->k_c1_last_core;
	int k_rem = x->k_rem;
	int p_l = x->p_l, m_pad = x->m_pad, k_pad = x->k_pad, n_pad = x->n_pad;
	int Mb = x->Mb, Kb = x->Kb, Nb = x->Nb;
	int M_padded = x->M_padded;

	int m, k, n, m_start, m_end, m_inc, k_start, k_end, k_inc;
	int k_cb, n_c_t, m_c_t, p_used, core;

	for(n = 0; n < Nb; n++) {

		if(n % 2) {
			k_start = Kb - 1;
			k_end = -1;
			k_inc = -1;
		} else {
			k_start = 0;
			k_end = Kb;
			k_inc = 1;
		}

		n_c_t = n_c;
		if((n == Nb - 1) && n_pad) {
			n_c_t = n_c1;
		}

		for(k = k_start; k != k_end; k += k_inc) {

			if(n % 2) {
				if(k % 2) {
					m_start = 0;
					m_end = Mb;
					m_inc = 1;
				} else {
					m_start = Mb - 1;
					m_end = -1;
					m_inc = -1;
				}
			} else {
				if(k % 2) {
					m_start = Mb - 1;
					m_end = -1;
					m_inc = -1;
				} else {
					m_start = 0;
					m_end = Mb;
					m_inc = 1;
				}
			}

			if((k == Kb - 1) && k_pad) {
				p_used = p_l;
				k_cb = k_rem; 
			} else {
				p_used = p;
				k_cb = p_used*k_c;
			}

			#pragma omp parallel for private(m, m_c_t, core)
			for(int m_ind = 0; m_ind < Mb; m_ind++) {
			// for(m = 0; m < Mb; m++) {

				m = m_start + m_ind*m_inc;

				m_c_t = m_c; 
				if((m == Mb - 1) && m_pad) {
					m_c_t = m_c1;
				}

				// #pragma omp parallel for private(core)
				for(core = 0; core < p_used; core++) {

					// These vars must be private to thread, 
					// otherwise out of bounds memory access possible
					int k_c_t, k_c_x, n_reg, m_reg;

					if((k == Kb - 1) && k_pad) {
						k_c_t = (core == (p_l - 1) ? k_c1_last_core : k_c1);
						k_c_x = k_c1;
					} else {
						k_c_t = k_c;
						k_c_x = k_c;
					}

					int a_ind = k*M_padded*p*k_c + m*m_c*k_cb + core*m_c_t*k_c_x;
					int b_ind = n*K*n_c + k*p*k_c*n_c_t + core*k_c_x*n_c_t;
					int c_ind = n*M_padded*n_c + m*m_c*n_c_t;

					for(n_reg = 0; n_reg < (n_c_t / n_r); n_reg++) {
						for(m_reg = 0; m_reg < (m_c_t / m_r); m_reg++) {	

							cake_sgemm_ukernel(&A_p[a_ind + m_reg*m_r*k_c_t], 
											&B_p[b_ind + n_reg*k_c_t*n_r], 
											&C_p[c_ind + n_reg*m_c_t*n_r + m_reg*m_r*n_r], 
											m_r, n_r, k_c_t, cake_cntx);
						}
					}
				}
			}
		}
	}
}


