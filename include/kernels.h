#ifndef KERNELS_H_
#define KERNELS_H_

#include <arm_neon.h>

#define MR_FACT 2
#define NR_FACT 12
#define MR_MIN 8
#define NR_MIN 12


// transpose functions
void transpose_mxn(float* A, float* B, int lda, int ldb, int m, int n);
void transpose_8x8(float* A, float* B, int lda, int ldb);


typedef void cake_sgemm_armv8(float* A, float* B, float* C, int m, int n, int k);

void cake_sgemm_armv8_8x12(float* A, float* B, float* C, int m, int n, int k);
									
void cake_sgemm_armv8_8x24(float* A, float* B, float* C, int m, int n, int k);
									
void cake_sgemm_armv8_10x12(float* A, float* B, float* C, int m, int n, int k);
									
void cake_sgemm_armv8_10x24(float* A, float* B, float* C, int m, int n, int k);
									
static cake_sgemm_armv8* kernel_map[2][2] = 
{
	
	{cake_sgemm_armv8_8x12,cake_sgemm_armv8_8x24},
	{cake_sgemm_armv8_10x12,cake_sgemm_armv8_10x24}
};

void matmul_8x12_sparse(float* A, float* B, float* C, int childnya, int fin);


/*
Fused softmax and matmul: S = softmax(L) (logits), A = S*V
Online softmax: updates running max and denominator for each row of A.
Processes one 8x12 tile of L (logits) at a time.
*/
void softmax_matmul_fused_armv8_8x12(
        float       *S_tile,       // actual_rows_S × actual_cols_S logits (row-major, with ld_S)
        const float *V_panel,      // actual_cols_S × d_v (row-major, with ld_V)
        float       *A_panel,      // actual_rows_S × d_v output block (row-major, with ld_A)
        float       *row_max_vec,  // actual_rows_S running maxima
        float       *row_den_vec,  // actual_rows_S running denominators
        int          d_v,          // value dimension, multiple of 4
        int          actual_rows_S, // Actual number of rows in S_tile (e.g., ny_c_t)
        int          actual_cols_S, // Actual number of columns in S_tile (e.g., nx_c_t)
        int          ld_S,          // Leading dimension of S_tile (allocated width, e.g., x->n_c)
        int          ld_V,          // Leading dimension of V_panel (usually d_v)
        int          ld_A           // Leading dimension of A_panel (usually d_v)
);

#endif // KERNELS_H_ 