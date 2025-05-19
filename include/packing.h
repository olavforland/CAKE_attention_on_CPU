// #pragma once
#include "common.h"


// Dense Packing

void pack_C(float* C, float* C_p, int M, int N, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch) ;
void pack_B(float* B, float* B_p, int K, int N, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch) ;
double pack_A(float* A, float* A_p, int M, int K, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch) ;
void unpack_C(float* C, float* C_p, int M, int N, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch) ;


void pack_test1(float* A, float* A_p, int M, int K, int p, cake_cntx_t* cake_cntx);

double pack_A_single_buf_k_first(float* A, float* A_p, int M, int K, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
void pack_B_k_first(float* B, float* B_p, int K, int N, blk_dims_t* x, cake_cntx_t* cake_cntx);
void pack_C_single_buf_k_first(float* C, float* C_p, int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);

double pack_A_single_buf_k_first_blis(float* A, float* A_p, int M, int K, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
void pack_B_k_first_blis(float* B, float* B_p, int K, int N, blk_dims_t* x, cake_cntx_t* cake_cntx);



double pack_A_single_buf_m_first(float* A, float* A_p, int M, int K, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
void pack_B_m_first(float* B, float* B_p, int K, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
void pack_C_single_buf_m_first(float* C, float* C_p, int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);

double pack_A_single_buf_n_first(float* A, float* A_p, int M, int K, int p, blk_dims_t* x, cake_cntx_t* cake_cntx) ;
void pack_B_n_first(float* B, float* B_p, int K, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx) ;
void pack_C_single_buf_n_first(float* C, float* C_p, int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx) ;


void pack_ob_A_single_buf(float* A, float* A_p, int M, int K, int m1, 
				int m2, int m_c, int k_c, int m_r, bool pad);
void pack_ob_B_single_buf(float* B, float* B_p, int K, int N, int n1,
	            int k_c, int n_c, int n_r, bool pad_n) ;
void pack_ob_C_single_buf(float* C, float* C_p, int M, int N, int m1, int n1, int m2,
				int m_c, int n_c, int m_r, int n_r, bool pad_m, bool pad_n);


void pack_ob_A_multiple_buf(float* A, float* A_p, int M, int K, int m1, int k1, int m2, int m_c, int k_c, int m_r, bool pad) ;
void pack_ob_C_multiple_buf(float* C, float* C_p, int M, int N, int m1, int n1, int m2,
				int m_c, int n_c, int m_r, int n_r, bool pad);
void pack_ob_B_parallel(float* B, float* B_p, int K, int N, int n1,
            int k_c, int n_c, int n_r, bool pad_n);


void pack_ob_A_parallel(float* A, float* A_p, int M, int K, int m1, 
				int m2, int m_c, int k_c, int m_r, bool pad);


// unpacking
void unpack_ob_C_single_buf(float* C, float* C_p, int M, int N, int m1, int n1, int m2,
				int m_c, int n_c, int m_r, int n_r);
void unpack_ob_C_multiple_buf(float* C, float* C_p, int M, int N, int m1, int n1, int m2,
				int m_c, int n_c, int m_r, int n_r);

void unpack_C_single_buf_k_first(float* C, float* C_p, int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
void unpack_C_single_buf_m_first(float* C, float* C_p, int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
void unpack_C_single_buf_n_first(float* C, float* C_p, int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
void unpack_C_rsc(float* C, float** C_p, int M, int N, int m_c, int n_c, int n_r, int m_r, int p, int alpha_n);
void unpack_C_old(float* C, float** C_p, int M, int N, int m_c, int n_c, int n_r, int m_r, int p);



// helper funcs
size_t cake_sgemm_packed_A_size(int M, int K, int p, blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch);
size_t cake_sgemm_packed_B_size(int K, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx);
size_t cake_sgemm_packed_C_size(int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch);


void pack_A_mr_x_kc(float* A, float* A_p, int K, int k_c, int m_r);
void pack_B_nr_x_kc(float* B, float* B_p, int N, int k_c, int n_r);





// kernel helper functions
inline void A_packing_kernel
(       
	int               cdim0,
	int               k0,
	float*      kappa,
	float*      a, int inca0, int lda0,
	float*      p,              int ldp0
) {


pack_A_mr_x_kc(a, p, inca0, k0, cdim0);

}



inline void B_packing_kernel
(       
	int               cdim0,
	int               k0,
	float*      kappa,
	float*      a, int inca0, int lda0,
	float*      p,              int ldp0
) {


pack_B_nr_x_kc(a, p, lda0, k0, cdim0);

}




