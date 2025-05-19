#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstring>
#include <cblas.h>
#include <chrono>
#include <omp.h>
#include <algorithm>                  // for std::max
#include <iomanip>                    // Required for std::fixed and std::setprecision
#include "cake.h"
// #include "attention.h"
#include <unistd.h>
#include <sys/resource.h>

void rand_init(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
    }
}

void fixed_init(float* mat, int rows, int cols, float val) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = val;
    }
}
void increment_init(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = i;
    }
}



void print_matrix(float* mat, int rows, int cols) {
    // show + on positives, fixed point with 2 decimals
    std::cout << std::showpos << std::fixed << std::setprecision(2);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << mat[i * cols + j] << " ";
        }
        std::cout << "\n";
    }

    // reset to default float formatting and hide the + again
    std::cout << std::noshowpos << std::defaultfloat;
}


// C:   pointer to M×N row-major matrix
// M,N: dimensions
// ldC: leading dimension (here ldC == N)
// threads: # of OpenMP threads
void rowwise_softmax(float* R, int N, int M, int ldC, int threads) {
    omp_set_num_threads(threads);
    #pragma omp parallel for
    for(int i = 0; i < N; ++i) {
        float* row = R + i * ldC;


        float maxv = -INFINITY;
        float max_prev;
        float norm = 0.0f;

        // online softmax
        for(int j = 0; j < M; ++j) {
            max_prev = maxv;
            maxv = std::max(max_prev, row[j]);
            norm = norm * std::expf(max_prev - maxv) + std::expf(row[j] - maxv);
        }

        // normalize
        #pragma omp simd
        for(int j = 0; j < M; ++j)
            row[j] = std::expf(row[j] - maxv) / norm;
    }
}
void compare_results(float* A, float* B, int len) {
    float max_diff = 0.0f;
    for (int i = 0; i < len; ++i) {
        float diff = std::abs(A[i] - B[i]);
        if (diff > max_diff) max_diff = diff;
    }
    std::cout << "Max absolute difference: " << max_diff << std::endl;
}

#include <iostream>
#include <vector>
#include <cmath>
#include <utility>  // for std::pair

#include <iostream>
#include <iomanip>
#include <cmath>

void compare_results_and_get_wrong_positions(const float* A,
                     const float* B,
                     int rows,
                     int cols,
                     float tolerance = 0.0f)
{
    float max_diff = 0.0f;
    int max_row = 0, max_col = 0;

    // First pass: find max diff & its location
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            float diff = std::abs(A[idx] - B[idx]);
            if (diff > max_diff) {
                max_diff = diff;
                max_row  = i;
                max_col  = j;
            }
        }
    }

    // Print the matrix of results
    std::cout << "Comparison matrix (tolerance = " << tolerance << "):\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            float a = A[idx], b = B[idx];
            float diff = std::abs(a - b);

            if (diff > tolerance) {
                // wrong entry: got/expected
                std::cout 
                    << std::setw(10) 
                    << std::fixed << std::setprecision(2)
                    << a << "/" << b;
            }
            else {
                // within tolerance: just print got
                std::cout 
                    << std::setw(10) 
                    << std::fixed << std::setprecision(2)
                    << a;
            }

            if (j < cols - 1) std::cout << " ";
        }
        std::cout << "\n";
    }

    // Summary of maximum discrepancy
    std::cout << "\nMax absolute difference: " 
              << std::fixed << std::setprecision(2)
              << max_diff
              << " at (row=" << max_row 
              << ", col=" << max_col << ")\n";
}




int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " M K N threads trials\n";
        return 1;
    }

    int N = std::atoi(argv[1]);
    int d = std::atoi(argv[2]);
    int threads = std::atoi(argv[3]);
    int trials = std::atoi(argv[4]);
    std::cout << "N = " << N << ", d = " << d << ", threads = " << threads << ", trials = " << trials << std::endl;

    

    cake_cntx_t* cake_cntx = cake_query_cntx();
    // Restore cake_cntx initialization
    cake_cntx->m_map = 0;
    cake_cntx->n_map = 0;

    double materialized_total = 0.0;
    double fused_total = 0.0;

    float *KT = (float*) malloc(N * d * sizeof(float));
    float *Q = (float*) malloc(N * d * sizeof(float));
    float *V = (float*) malloc(N * d * sizeof(float));
    float *S_materialized = (float*) calloc(N * N,  sizeof(float));
    float *A_materialized = (float*) calloc(N * d, sizeof(float));
    float *S_fused = (float*) calloc(N * N,  sizeof(float));
    float *A_fused = (float*) calloc(N * d, sizeof(float));

    long trial_mem[trials];
    for (int i = 0; i < trials; ++i) {
        rand_init(Q, N, d);
        rand_init(KT, d, N);
        rand_init(V, N, d);
        std::memset(S_materialized, 0, N * N * sizeof(float));
        std::memset(A_materialized, 0, N * d * sizeof(float));
        std::memset(S_fused, 0, N * N * sizeof(float));
        std::memset(A_fused, 0, N * d * sizeof(float));

        auto start_materialized = std::chrono::high_resolution_clock::now();
        cake_sgemm(Q, KT, S_materialized, N, N, d, threads, cake_cntx, NULL, 0, 0, 1, 0, KMN);
        rowwise_softmax(S_materialized, N, N, N, threads);
        cake_sgemm(S_materialized, V, A_materialized, N, d, N, threads, cake_cntx, NULL, 0, 0, 1, 0, KMN);
        auto end_materialized = std::chrono::high_resolution_clock::now();
        materialized_total += std::chrono::duration<double>(end_materialized - start_materialized).count();
        
        auto start_fused = std::chrono::high_resolution_clock::now();
        cake_attention(Q, KT, V, S_fused, A_fused, N, d, threads, cake_cntx, NULL, 1.0f, 0.0f, KMN, 0, 0);
        auto end_fused = std::chrono::high_resolution_clock::now();
        fused_total += std::chrono::duration<double>(end_fused - start_fused).count();
    }

        // printf("A_materialized:\n");
        // print_matrix(A_materialized, 8, d);
        // printf("A_fused:\n");
        // print_matrix(A_fused, 8, d);

    // Compare results
    compare_results(A_materialized, A_fused, N * d);
    printf("Materialized time: %f\n", materialized_total);
    printf("Fused time: %f\n", fused_total);

    free(KT);
    free(Q);
    free(V);
    free(S_materialized);
    free(S_fused);
    free(A_materialized);
    free(A_fused);
    free(cake_cntx);

    return 0;
}
