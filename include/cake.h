#pragma once

#include "kernel_wrapper.h"
#include "util.h"
#include "tiling.h"
#include "packing.h"
// #include "kernels.h"
// #include "autotune.h"


void schedule_attention(float* Q_p, float* KT_p, float* S, float** S_p, float* V, float* A, float** A_p, int N, int d_hid, int p, cake_cntx_t* cake_cntx, blk_dims_t* x);
double cake_attention(float* Q, float* KT, float* V, float* S, float* A, int N, int d, int p, 
                        cake_cntx_t* cake_cntx, char* argv[],
                        float alpha, float beta, enum sched sch, int ncu=0, int dcu=0);

// Dense MM scheduling
double cake_sgemm(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, char* argv[] = NULL, bool packedA = 0, bool packedB = 0, 
	float alpha = 1, float beta = 0, enum sched sch = NA, int mcu = 0, int kcu = 0, int ncu = 0);
void schedule(float* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x, enum sched sch, bool sparse, bool small);
void schedule_KMN(float* A_p, float* B_p, float* C, float** C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x) ;
void schedule_MKN(float* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);
void schedule_NKM(float* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);



void schedule_KMN_online(float* A, float* B, float* C, float** A_p, float* B_p, float** C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);
void schedule_KMN_C_unpacked(float* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);
void schedule_KMN_2d(float* A, float* B, float* C, float* A_p, float* B_p, float** C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);
void schedule_KMN_2d_small(float* A, float* B, float* C, float* A_p, float* B_p, float** C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);


double cake_sgemm_online(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, char* argv[] = NULL, bool packedA = 0, bool packedB = 0, 
	float alpha = 1, float beta = 0, enum sched sch = NA, int mcu = 0, int kcu = 0, int ncu = 0);
double cake_sgemm_test(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, char* argv[] = NULL, bool packedA = 0, bool packedB = 0, 
	float alpha = 1, float beta = 0, enum sched sch = NA);
double cake_sgemm_2d(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, char* argv[] = NULL, bool packedA = 0, bool packedB = 0, 
	float alpha = 1, float beta = 0, enum sched sch = NA, int mcu = 0, int kcu = 0, int ncu = 0);
double cake_sgemm_2d_small(float* A, float* B, float* C, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, char* argv[] = NULL, bool packedA = 0, bool packedB = 0, 
	float alpha = 1, float beta = 0, enum sched sch = NA);



// small matrix handling
bool cake_gemm_small(float* A, float* B, float* C, int M, int N, int K, int p, 
	blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch, char* argv[] = NULL);



void schedule_KMN_small(float* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);
void schedule_MKN_small(float* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);
void schedule_NKM_small(float* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);

void schedule_NKM_small_A_packed(float* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);
void schedule_MKN_small_B_packed(float* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);
void schedule_KMN_small_C_packed(float* A_p, float* B_p, float* C_p, int M, int N, int K, int p, 
	cake_cntx_t* cake_cntx, blk_dims_t* x);

bool cake_gemm_small(float* A, float* A_p, float* B, float* B_p, float* C, float* C_p, 
	int M, int N, int K, int p, blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch);


// choose cake schedule based on M,N,K values
enum sched set_schedule(enum sched sch, int M, int N, int K);




