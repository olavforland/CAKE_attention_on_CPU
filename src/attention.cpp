#include "cake.h"
#include <algorithm> // for std::min

// Remove the min macro since we're using std::min
// #ifndef min
// #define min(a,b) ((a) < (b) ? (a) : (b))
// #endif

// based on schedule KMN
void schedule_attention(float* Q_p, float* KT_p, float* S, float** S_p, float* V_p, float*A, float** A_p, int N, int d_hid, int p, cake_cntx_t* cake_cntx, blk_dims_t* x) {

	// copy over block dims to local vars to avoid readibility ussiues with x->
	int ny_r = cake_cntx->mr, nx_r = cake_cntx->nr;
	int ny_map = cake_cntx->m_map, nx_map = cake_cntx->n_map;

	int ny_c = x->m_c, d_c = x->k_c, nx_c = x->n_c;
	int ny_c1 = x->m_c1, d_c1 = x->k_c1, nx_c1 = x->n_c1;
	int ny_c1_last_core = x->m_c1_last_core;
	int nyr_rem = x->mr_rem;
	int p_l = x->p_l, ny_pad = x->m_pad, d_pad = x->k_pad, nx_pad = x->n_pad;
	int Nyb = x->Mb, Nxb = x->Nb, db = x->Kb;
	int Ny_padded = x->M_padded, Nx_padded = x->N_padded;

	// // Print key parameters to understand difference between N=190 and N=192
	// printf("Basic dims: ny_r=%d, nx_r=%d, ny_map=%d, nx_map=%d\n", ny_r, nx_r, ny_map, nx_map);
	// printf("Cache blocks: ny_c=%d, d_c=%d, nx_c=%d\n", ny_c, d_c, nx_c);
	// printf("Last blocks: ny_c1=%d, d_c1=%d, nx_c1=%d, ny_c1_last_core=%d\n", ny_c1, d_c1, nx_c1, ny_c1_last_core);
	// printf("Padding: ny_pad=%d, d_pad=%d, nx_pad=%d, nyr_rem=%d\n", ny_pad, d_pad, nx_pad, nyr_rem);
	// printf("Block counts: Nyb=%d, Nxb=%d, db=%d, p_l=%d\n", Nyb, Nxb, db, p_l);
	// printf("Padded dims: Ny_padded=%d, Nx_padded=%d\n", Ny_padded, Nx_padded);
	// printf("N=%d, d_hid=%d, cores p=%d\n", N, d_hid, p);
	
	// // Print critical dimensions for this matrix size
	// printf("Tile size: ny_r × nx_r = %d × %d\n", ny_r, nx_r);
	// printf("Full tile count: (N / ny_r)=%d, (N / nx_r)=%d\n", N / ny_r, N / nx_r);
	// printf("Remainder: (N %% ny_r)=%d, (N %% nx_r)=%d\n", N % ny_r, N % nx_r);
	// printf("CRITICAL CHECK: nx_c=%d, (N %% nx_c)=%d\n", nx_c, N % nx_c);
	// printf("CRITICAL CHECK: ny_c=%d, (N %% (p*ny_c))=%d\n", ny_c, N % (p*ny_c));


	int ny1, nx1, ny, d, nx, ny_start, ny_end, ny_inc, d_start, d_end, d_inc, S_offset = 0;
	int ny_cb, nx_c_t, p_used, core;

	// for online computation of attention
	float row_max[N], row_den[N];
	for (int i = 0; i < N; i++) {
		row_max[i] = -INFINITY;
		row_den[i] = 0.0;
	}

	// Add debug for first nx block analysis
	if ((0 == Nxb - 1) && nx_pad) {
		nx_c_t = nx_c1;
		nx1 = (N - (N % nx_c));
	} else {
		nx_c_t = nx_c;
		nx1 = 0*nx_c;
	}
	
	// Add debug for first ny block analysis
	if ((0 == Nyb - 1) && ny_pad) {
		p_used = p_l;
		ny_cb = ny_r*nyr_rem;
		ny1 = (N - (N % (p*ny_c)));
	} else {
		p_used = p;
		ny_cb = p_used*ny_c;
		ny1 = 0*p*ny_c;
	}

	for(nx = 0; nx < Nxb; nx++) {
		// Clear S_p buffers at the start of each nx block to avoid using stale data
		#pragma omp parallel for private(core)
		for(core = 0; core < p; core++) {
			memset(S_p[core], 0, x->m_c * x->n_c * sizeof(float));
			
			// Also clear A_p if this is the first nx block
			if (nx == 0) {
				memset(A_p[core], 0, x->m_c * d_hid * sizeof(float));
			}
		}

		if(nx % 2) {
			ny_start = Nyb - 1;
			ny_end = -1;
			ny_inc = -1;
		} else {
			ny_start = 0;
			ny_end = Nyb;
			ny_inc = 1;
		}

		if((nx == Nxb - 1) && nx_pad) {
			nx_c_t = nx_c1;
			nx1 = (N - (N % nx_c));
			// If N is divisible by nx_c, we need to be careful with nx1
			if (N % nx_c == 0 && N >= 192) {
				nx1 = N - nx_c;  // Fix for dimensions divisible by block size
			}
		} else {
			nx_c_t = nx_c;
			nx1 = nx*nx_c;
		}

		for(ny = ny_start; ny != ny_end; ny += ny_inc) {
			if(nx % 2) {
				if(ny % 2) {
					d_start = 0;
					d_end = db;
					d_inc = 1;
				} else {
					d_start = db - 1;
					d_end = -1;
					d_inc = -1;
				}
			} else {
				if(ny % 2) {
					d_start = db - 1;
					d_end = -1;
					d_inc = -1;
				} else {
					d_start = 0;
					d_end = db;
					d_inc = 1;
				}
			}

			if((ny == Nyb - 1) && ny_pad) {
				p_used = p_l;
				ny_cb = ny_r*nyr_rem ; //M % (p*m_c);
				ny1 = (N - (N % (p*ny_c)));
			} else {
				p_used = p;
				ny_cb = p_used*ny_c;
				ny1 = ny*p*ny_c; // Make sure we're calculating ny1 correctly
			}

			// A_p should NOT be cleared - it accumulates results

			// pragma omp here (i_c loop)
			#pragma omp parallel for private(core,d)
			for(core = 0; core < p_used; core++) {

				// These vars must be private to thread, 
				// otherwise out of bounds memory access possible
				int ny_c_t, ny_c_x, d_c_t, nx_reg, ny_reg;

				if((ny == Nyb - 1) && ny_pad) {
					ny_c_t = (core == (p_l - 1) ? ny_c1_last_core : ny_c1);
					ny_c_x = ny_c1;
				} else {
					ny_c_t = ny_c;
					ny_c_x = ny_c; 
				}
				
				// pragma omp also here possible (j_r loop)
				for(nx_reg = 0; nx_reg < (nx_c_t / nx_r); nx_reg++) {
					for(ny_reg = 0; ny_reg < (ny_c_t / ny_r); ny_reg++) {	
						// Calculate a micro-tile of QK^T
						for(d = d_start; d != d_end; d += d_inc) {
							d_c_t = d_c; 
							if((d == db - 1) && d_pad) {
								d_c_t = d_c1;
							}
						
							int q_ind = ny1*d_hid + d*ny_cb*d_c + core*ny_c_x*d_c_t;
							int kt_ind = nx*d_hid*nx_c + d*d_c*nx_c_t;

							// Compact 8×12 scratch tiles laid out consecutively (row-stride = 12)
							int tiles_per_row = nx_c_t / nx_r;            // e.g. 24/12 = 2 when N=192
							size_t tile_id    = ny_reg * tiles_per_row + nx_reg;
							size_t S_tile_offset_compact = tile_id * ny_r * nx_r; // 96 floats per tile

							kernel_map[ny_map][nx_map](
								&Q_p[q_ind + ny_reg*ny_r*d_c_t],
								&KT_p[kt_ind + nx_reg*d_c_t*nx_r],
								&S_p[core][S_tile_offset_compact],
								ny_r, nx_r, d_c_t);
						}
						
	
						// Default values for a full tile
						int actual_rows_S = ny_r;
						int actual_cols_S = nx_r;
						
						// If this is the last row tile, adjust actual_rows_S
						if ((ny == ny_end - ny_inc) && (ny_reg == (ny_c_t / ny_r) - 1) && (N % ny_r != 0)) {
							actual_rows_S = N % ny_r;
						}

						// Global row index of the first row of this micro-tile
						int row_base = ny1 + core * ny_c_x + ny_reg * ny_r;

						// If the tile starts beyond the matrix, skip it entirely
						if (row_base >= N) {
							continue;
						}

						// Clamp rows when the tile straddles the bottom edge
						if (row_base + actual_rows_S > N) {
							actual_rows_S = N - row_base;
						}

						// Recompute compact-tile offset (same formula as for kernel write)
						int tiles_per_row = nx_c_t / nx_r;
						size_t tile_id    = ny_reg * tiles_per_row + nx_reg;
						size_t S_tile_offset = tile_id * ny_r * nx_r; // 8×12 block start

						// ------------------------------------------------------------------
						// DEBUG: dump the first two tiles for N==192 to understand duplication.
						// ------------------------------------------------------------------
						// static int dbg_tiles_dumped = 0;
						// if(N == 192 && dbg_tiles_dumped < 2 && core == 0 && nx == 0 && ny == 0) {
						//     printf("\n--- DEBUG TILE %d (ny_reg=%d nx_reg=%d) ---\n", dbg_tiles_dumped, ny_reg, nx_reg);
						//     for(int rdbg = 0; rdbg < ny_r; ++rdbg) {
						//         printf("row %d :", rdbg);
						//         for(int cdbg = 0; cdbg < nx_r; ++cdbg) {
						//             float val_dbg = S_p[core][S_tile_offset + rdbg*nx_r + cdbg];
						//             printf(" %+.4f", val_dbg);
						//         }
						//         printf("\n");
						//     }
						//     // dump the first element of each Q row feeding this tile
						//     int q_ind_dbg = ny1*d_hid + d_start*ny_cb*d_c + core*ny_c_x*d_c_t;
						//     float *Qrow0 = &Q_p[q_ind_dbg + ny_reg*ny_r*d_c_t];
						//     printf("Q rows first val :");
						//     for(int rdbg = 0; rdbg < ny_r; ++rdbg) {
						//         printf(" %+.4f", Qrow0[rdbg*d_c_t]);
						//     }
						//     printf("\n");
						//     dbg_tiles_dumped++;
						// }
						// ------------------------------------------------------------------
						// Zero logits tile BEFORE accumulating Q*K^T over d_c_t columns. (disabled for now)
						// ------------------------------------------------------------------
						// memset(&S_p[core][S_tile_offset], 0, ny_r * nx_r * sizeof(float));

						size_t V_panel_offset = (nx * nx_c + nx_reg * nx_r) * d_hid;

						size_t A_panel_offset = row_base * d_hid;
						int    row_idx_base = row_base;

						// Call the softmax kernel (after GEMM accumulation over d)
						softmax_matmul_fused_armv8_8x12(
							&S_p[core][S_tile_offset],
							&V_p[V_panel_offset],
							&A_p[core][A_panel_offset],
							&row_max[row_idx_base],
							&row_den[row_idx_base],
							d_hid,
							actual_rows_S,
							actual_cols_S,
							nx_r,     // ld_S = 12 (compact tile stride)
							d_hid,   // Leading dimension of V
							d_hid    // Leading dimension of A
						);
					}
				}
			}

			S_offset = ny*p*ny_c*N + nx*nx_c;

			// Unpack A_p to A (only for valid rows)
			#pragma omp parallel for private(core)
			for(core = 0; core < p_used; core++) {
				int ny_c_t, ny_c_x;

				if((ny == Nyb - 1) && ny_pad) {
					ny_c_t = (core == (p_l - 1) ? ny_c1_last_core : ny_c1);
					ny_c_x = ny_c1;
				} else {
					ny_c_t = ny_c;
					ny_c_x = ny_c;
				}

				// Unpack A_p to A - each micro-row is d_hid wide
				for(int i = 0; i < ny_c_t; i++) {
					int global_row = ny1 + core*ny_c_x + i;
					if(global_row >= N) break; // don't write padded rows
					memcpy(&A[global_row * d_hid],
						   &A_p[core][i * d_hid],
						   d_hid * sizeof(float));
				}
			}
		}
	}
}


double cake_attention(float* Q, float* KT, float* V, float* S, float* A, int N, int d, int p, 
	cake_cntx_t* cake_cntx, char* argv[],
	float alpha, float beta, enum sched sch, int ncu, int dcu) {

	blk_dims_t* x = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	if (!x) {
		printf("Error: malloc failed for x\n");
		exit(1);
	}

	omp_set_num_threads(p);

	size_t Q_sz, KT_sz;
	struct timespec start, end, start1, end1;
	long seconds, nanoseconds;
	double diff_t;
	float *Q_p, *KT_p;

	clock_gettime(CLOCK_REALTIME, &start1);

	init_block_dims(N, N, d, p, x, cake_cntx, sch, argv, 4, ncu, dcu, ncu);
	sch = x->sch;

	blk_dims_t* x_for_V = (blk_dims_t*) malloc(sizeof(blk_dims_t));
	if (!x_for_V) {
		printf("Error: malloc failed for x_for_V\n");
		free(x);
		exit(1);
	}

	init_block_dims(N, N, d, p, x_for_V, cake_cntx, sch, argv, 4, ncu, dcu, ncu);

	// Pack Q (A matrix in QK^T)
	clock_gettime(CLOCK_REALTIME, &start);
	Q_sz = cake_sgemm_packed_A_size(N, d, p, x, cake_cntx, sch);
	if(posix_memalign((void**) &Q_p, 64, Q_sz)) {
		printf("Error: posix_memalign failed for Q_p\n");
		free(x); 
		free(x_for_V);
		exit(1);
	}
	pack_A(Q, Q_p, N, d, p, x, cake_cntx, sch);

	// Pack KT (B matrix in QK^T)
	KT_sz = cake_sgemm_packed_B_size(d, N, p, x, cake_cntx);
	if(posix_memalign((void**) &KT_p, 64, KT_sz)) {
		printf("Error: posix_memalign failed for KT_p\n");
		free(Q_p);
		free(x);
		free(x_for_V);
		exit(1);
	}
	pack_B(KT, KT_p, d, N, p, x, cake_cntx, sch);

	// Setup S and A buffers
	float *S_p_buffers[p];
	float *A_p_buffers[p];

	for(int i = 0; i < p; i++) {
		S_p_buffers[i] = (float*) calloc(x->m_c * x->n_c, sizeof(float));
		A_p_buffers[i] = (float*) calloc(x->m_c * d, sizeof(float));
		if(!S_p_buffers[i] || !A_p_buffers[i]) {
			printf("Error: calloc failed for S_p_buffers or A_p_buffers\n");
			free(Q_p);
			free(KT_p);
			for(int j = 0; j < i; j++) {
				free(S_p_buffers[j]);
				free(A_p_buffers[j]);
			}
			free(x);
			free(x_for_V);
			exit(1);
		}
	}

	// Run attention computation
	clock_gettime(CLOCK_REALTIME, &start);
	schedule_attention(Q_p, KT_p, S, S_p_buffers, V, A, A_p_buffers, N, d, p, cake_cntx, x); 
	clock_gettime(CLOCK_REALTIME, &end);
	
	// Cleanup
	for(int i = 0; i < p; i++) {
		free(S_p_buffers[i]);
		free(A_p_buffers[i]);
	}
	free(Q_p);
	free(KT_p);
	free(x_for_V);
	free(x);

	// Calculate time
	seconds = end.tv_sec - start.tv_sec;
	nanoseconds = end.tv_nsec - start.tv_nsec;
	diff_t = seconds + nanoseconds*1e-9;
	return diff_t;
}
