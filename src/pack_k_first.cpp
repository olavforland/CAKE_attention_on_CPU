#include "cake.h"




// pack the entire matrix A into a single contiguous cache-aligned buffer
double pack_A_single_buf_k_first(float* A, float* A_p, int M, int K, int p, blk_dims_t* x, cake_cntx_t* cake_cntx) {
   
   // copy over block dims to local vars to avoid readibility ussiues with x->
   int m_r = cake_cntx->mr;

   int m_c = x->m_c, k_c = x->k_c;
   int m_c1 = x->m_c1, k_c1 = x->k_c1;
   int m_c1_last_core = x->m_c1_last_core;
   int mr_rem = x->mr_rem;
   int p_l = x->p_l, m_pad = x->m_pad, k_pad = x->k_pad;
   int Mb = x->Mb, Kb = x->Kb;

   struct timespec start, end;
   double diff_t;

   int m, k, A_offset = 0, A_p_offset = 0;
   int m_cb, k_c_t, p_used, core;

   clock_gettime(CLOCK_REALTIME, &start);

   for(m = 0; m < Mb; m++) {

      if((m == Mb - 1) && m_pad) {
         p_used = p_l;
         m_cb = m_r*mr_rem ; 
      } else {
         p_used = p;
         m_cb = p_used*m_c;
      }

      for(k = 0; k < Kb; k++) {
         
         k_c_t = k_c; 
         if((k == Kb - 1) && k_pad) {
            k_c_t = k_c1;
         }
         // m*m_c is the row number, K is number of columns per row, and p is the number of elements we process each time, and k*k_c is just column number
         A_offset = m*p*m_c*K + k*k_c;

         #pragma omp parallel for private(core)
         for(core = 0; core < p_used; core++) {

            int m_c_t, m_c_x;
            bool pad;

            if((m == Mb - 1) && m_pad) {
               m_c_t = (core == (p_l - 1) ? m_c1_last_core : m_c1);
               m_c_x = m_c1;
               pad = (core == (p_l - 1) ? 1 : 0);
            } else {
               m_c_t = m_c;
               m_c_x = m_c;
               pad = 0;
            }

            pack_ob_A_single_buf(&A[A_offset + core*m_c_x*K], &A_p[A_p_offset + core*m_c_x*k_c_t], 
               M, K, m*p*m_c, core*m_c_x, m_c_t, k_c_t, m_r, pad);
         }

         A_p_offset += m_cb*k_c_t;
      }
   }

     clock_gettime(CLOCK_REALTIME, &end);
     long seconds = end.tv_sec - start.tv_sec;
     long nanoseconds = end.tv_nsec - start.tv_nsec;
     diff_t = seconds + nanoseconds*1e-9;

     return diff_t;
}






void pack_test1(float* A, float* A_p, int M, int K, int p, cake_cntx_t* cake_cntx) {

   float kappa = 1.0;
   int lda = 1;

   #pragma omp parallel for
   for(int m3 = 0; m3 < M; m3 += 6) {
      A_packing_kernel(6, K, &kappa, &A[m3*K], K, lda, 
         &A_p[m3*K], 6);
   }     
}


// pack the entire matrix A into a single contiguous cache-aligned buffer using blis packing kernels
double pack_A_single_buf_k_first_blis(float* A, float* A_p, int M, int K, int p, blk_dims_t* x, cake_cntx_t* cake_cntx) {
   
   // copy over block dims to local vars to avoid readibility ussiues with x->
   int m_r = cake_cntx->mr;

   int m_c = x->m_c, k_c = x->k_c;
   int m_c1 = x->m_c1, k_c1 = x->k_c1;
   int m_c1_last_core = x->m_c1_last_core;
   int mr_rem = x->mr_rem;
   int p_l = x->p_l, m_pad = x->m_pad, k_pad = x->k_pad;
   int Mb = x->Mb, Kb = x->Kb;

   struct timespec start, end;
   double diff_t;

   int m, k, A_offset = 0, A_p_offset = 0;
   int m_cb, k_c_t, p_used, core;

   clock_gettime(CLOCK_REALTIME, &start);

   for(m = 0; m < Mb; m++) {

      if((m == Mb - 1) && m_pad) {
         p_used = p_l;
         m_cb = m_r*mr_rem ; 
      } else {
         p_used = p;
         m_cb = p_used*m_c;
      }

      for(k = 0; k < Kb; k++) {
         
         k_c_t = k_c; 
         if((k == Kb - 1) && k_pad) {
            k_c_t = k_c1;
         }

         A_offset = m*p*m_c*K + k*k_c;

         #pragma omp parallel for private(core)
         for(core = 0; core < p_used; core++) {

            int m_c_t, m_c_x;
            bool pad;

            if((m == Mb - 1) && m_pad) {
               m_c_t = (core == (p_l - 1) ? m_c1_last_core : m_c1);
               m_c_x = m_c1;
               pad = (core == (p_l - 1) ? 1 : 0);
            } else {
               m_c_t = m_c;
               m_c_x = m_c;
               pad = 0;
            }

            if(pad) {
               pack_ob_A_single_buf(&A[A_offset + core*m_c_x*K], &A_p[A_p_offset + core*m_c_x*k_c_t], 
                  M, K, m*p*m_c, core*m_c_x, m_c_t, k_c_t, m_r, pad);
           } else {          


            float kappa = 1.0;
            int lda = 1;

            for(int m3 = 0; m3 < m_c_t; m3 += m_r) {
               A_packing_kernel(m_r, k_c_t, &kappa, &A[A_offset + core*m_c_x*K + m3*K], K, lda, 
                  &A_p[A_p_offset + core*m_c_x*k_c_t + m3*k_c_t], m_r);
            }     

               // pack_ob_test(&A[A_offset + core*m_c_x*K], &A_p[A_p_offset + core*m_c_x*k_c_t], 
               //    M, K, m*p*m_c, core*m_c_x, m_c_t, k_c_t, m_r, pad);
           }
         }

         A_p_offset += m_cb*k_c_t;
      }
   }

     clock_gettime(CLOCK_REALTIME, &end);
     long seconds = end.tv_sec - start.tv_sec;
     long nanoseconds = end.tv_nsec - start.tv_nsec;
     diff_t = seconds + nanoseconds*1e-9;

     return diff_t;
}





void pack_B_k_first(float* B, float* B_p, int K, int N, blk_dims_t* x, cake_cntx_t* cake_cntx) {


   // copy over block dims to local vars to avoid readibility ussiues with x->
   int n_r = cake_cntx->nr;

   int k_c = x->k_c, n_c = x->n_c;
   int k_c1 = x->k_c1, n_c1 = x->n_c1;

   int k1, n1, n2;
   int ind1 = 0;

   int local_ind;

   // main portion of B that evenly fits into CBS blocks of size k_c x n_c 
   for(n1 = 0; n1 < (N - (N%n_c)); n1 += n_c) {
      
      #pragma omp parallel for private(k1,local_ind)
      for(k1 = 0; k1 < (K - (K%k_c)); k1 += k_c) {
         local_ind = 0;
         for(int n2 = 0; n2 < n_c; n2 += n_r) {
            for(int i = 0; i < k_c; i++) {
               for(int j = 0; j < n_r; j++) {
                  B_p[ind1 + local_ind + (k1/k_c)*k_c*n_c] = B[n1 + k1*N + n2 + i*N + j];
                  local_ind++;
               }
            }
         }
      }
      ind1 += ((K - (K%k_c))*n_c);

      k1 = (K - (K%k_c));
      if(k_c1) {

         #pragma omp parallel for private(n2,local_ind)
         for(n2 = 0; n2 < n_c; n2 += n_r) {
            local_ind = 0;
            for(int i = 0; i < k_c1; i++) {
               for(int j = 0; j < n_r; j++) {
                  B_p[ind1 + local_ind + n2*k_c1] = B[n1 + k1*N + n2 + i*N + j];
                  local_ind++;
               }
            }
         }
         ind1 += k_c1*n_c;
      }
   }

   // Process the final column of CBS blocks (sized k_c x n_c1) and perform N-dim padding 
   n1 = (N - (N%n_c));

   if(n_c1) {  

      #pragma omp parallel for private(k1,local_ind)
      for(k1 = 0; k1 < (K - (K%k_c)); k1 += k_c) {
         local_ind = 0;
         for(int n2 = 0; n2 < n_c1; n2 += n_r) {
            for(int i = 0; i < k_c; i++) {
               for(int j = 0; j < n_r; j++) {

                  if((n1 + n2 + j) >=  N) {
                     B_p[ind1 + local_ind + (k1/k_c)*k_c*n_c1] = 0.0;
                  } else {
                     B_p[ind1 + local_ind + (k1/k_c)*k_c*n_c1] = B[n1 + k1*N + n2 + i*N + j];
                  }

                  local_ind++;
               }
            }
         }
      }
      ind1 += ((K - (K%k_c))*n_c1);

      // Final CBS block (with k_c1 x n_c1 blocks) present in the lower right hand corner of B 
      k1 = (K - (K%k_c));
      if(k_c1) {

         #pragma omp parallel for private(n2,local_ind)
         for(int n2 = 0; n2 < n_c1; n2 += n_r) {
            local_ind = 0;
            for(int i = 0; i < k_c1; i++) {
               for(int j = 0; j < n_r; j++) {

                  if((n1 + n2 + j) >=  N) {
                     B_p[ind1 + local_ind + n2*k_c1] = 0.0;
                  } else {
                     B_p[ind1 + local_ind + n2*k_c1] = B[n1 + k1*N + n2 + i*N + j];
                  }

                  local_ind++;
               }
            }
         }
         ind1 += k_c1*n_c1;
      }
   }
}






void pack_B_k_first_blis(float* B, float* B_p, int K, int N, blk_dims_t* x, cake_cntx_t* cake_cntx) {


   // copy over block dims to local vars to avoid readibility ussiues with x->
   int n_r = cake_cntx->nr;

   int k_c = x->k_c, n_c = x->n_c;
   int k_c1 = x->k_c1, n_c1 = x->n_c1;

   int k1, n1, n2;
   int ind1 = 0;

   int local_ind;
   float kappa = 1.0;
   int lda = 1;

   // main portion of B that evenly fits into CBS blocks of size k_c x n_c 
   for(n1 = 0; n1 < (N - (N%n_c)); n1 += n_c) {
      
      #pragma omp parallel for private(k1,local_ind)
      for(k1 = 0; k1 < (K - (K%k_c)); k1 += k_c) {
         // local_ind = 0;

         for(int n2 = 0; n2 < n_c; n2 += n_r) {
            B_packing_kernel(n_r, k_c, &kappa, &B[n1 + k1*N + n2], lda, N, 
                  &B_p[ind1 + (k1/k_c)*k_c*n_c + n2*k_c], n_r);

            // B_packing_kernel(n_r, k_c, &kappa, &B[n1 + k1*N + n2], K, lda, 
            //       &B_p[ind1 + (k1/k_c)*k_c*n_c], n_r);

            // for(int i = 0; i < k_c; i++) {
            //    for(int j = 0; j < n_r; j++) {
            //       B_p[ind1 + local_ind + (k1/k_c)*k_c*n_c] = B[n1 + k1*N + n2 + i*N + j];
            //       local_ind++;
            //    }
            // }
         }
      }
      ind1 += ((K - (K%k_c))*n_c);

      k1 = (K - (K%k_c));
      if(k_c1) {

         #pragma omp parallel for private(n2,local_ind)
         for(n2 = 0; n2 < n_c; n2 += n_r) {
            // local_ind = 0;
            // B_packing_kernel(n_r, k_c1, &kappa, &B[n1 + k1*N + n2], K, lda, 
            //       &B_p[ind1 + n2*k_c1], n_r);
            B_packing_kernel(n_r, k_c1, &kappa, &B[n1 + k1*N + n2], lda, N, 
                  &B_p[ind1 + n2*k_c1], n_r);

            // for(int i = 0; i < k_c1; i++) {
            //    for(int j = 0; j < n_r; j++) {
            //       B_p[ind1 + local_ind + n2*k_c1] = B[n1 + k1*N + n2 + i*N + j];
            //       local_ind++;
            //    }
            // }
         }
         ind1 += k_c1*n_c;
      }
   }

   // Process the final column of CBS blocks (sized k_c x n_c1) and perform N-dim padding 
   n1 = (N - (N%n_c));

   if(n_c1) {  

      #pragma omp parallel for private(k1,local_ind)
      for(k1 = 0; k1 < (K - (K%k_c)); k1 += k_c) {
         local_ind = 0;
         for(int n2 = 0; n2 < n_c1; n2 += n_r) {
            for(int i = 0; i < k_c; i++) {
               for(int j = 0; j < n_r; j++) {

                  if((n1 + n2 + j) >=  N) {
                     B_p[ind1 + local_ind + (k1/k_c)*k_c*n_c1] = 0.0;
                  } else {
                     B_p[ind1 + local_ind + (k1/k_c)*k_c*n_c1] = B[n1 + k1*N + n2 + i*N + j];
                  }

                  local_ind++;
               }
            }
         }
      }
      ind1 += ((K - (K%k_c))*n_c1);

      // Final CBS block (with k_c1 x n_c1 blocks) present in the lower right hand corner of B 
      k1 = (K - (K%k_c));
      if(k_c1) {

         #pragma omp parallel for private(n2,local_ind)
         for(int n2 = 0; n2 < n_c1; n2 += n_r) {
            local_ind = 0;
            for(int i = 0; i < k_c1; i++) {
               for(int j = 0; j < n_r; j++) {

                  if((n1 + n2 + j) >=  N) {
                     B_p[ind1 + local_ind + n2*k_c1] = 0.0;
                  } else {
                     B_p[ind1 + local_ind + n2*k_c1] = B[n1 + k1*N + n2 + i*N + j];
                  }

                  local_ind++;
               }
            }
         }
         ind1 += k_c1*n_c1;
      }
   }
}



void pack_C_single_buf_k_first(float* C, float* C_p, int M, int N, int p, blk_dims_t* x, cake_cntx_t* cake_cntx) {


   // copy over block dims to local vars to avoid readibility ussiues with x->
   int m_r = cake_cntx->mr, n_r = cake_cntx->nr;

   int m_c = x->m_c, n_c = x->n_c;
   int m_c1 = x->m_c1, n_c1 = x->n_c1;
   int m_c1_last_core = x->m_c1_last_core;
   int mr_rem = x->mr_rem;
   int p_l = x->p_l, m_pad = x->m_pad, n_pad = x->n_pad;
   int Mb = x->Mb, Nb = x->Nb;

   int m, n, C_offset = 0, C_p_offset = 0;
   int m_cb, n_c_t, p_used, core;

   int m1, n1;
   bool pad_n;

   for(n = 0; n < Nb; n++) {

      if((n == Nb - 1) && n_pad) {
         n_c_t = n_c1;
         n1 = (N - (N % n_c));
         pad_n = 1;
      } else {
         n_c_t = n_c;
         n1 = n*n_c;
         pad_n = 0;
      }

      for(m = 0; m < Mb; m++) {

         if((m == Mb - 1) && m_pad) {
            p_used = p_l;
            m_cb = m_r*mr_rem ; 
            m1 = (M - (M % (p*m_c)));
         } else {
            p_used = p;
            m_cb = p_used*m_c;
            m1 = m*p*m_c;
         }

         C_offset = m*p*m_c*N + n*n_c;

         #pragma omp parallel for private(core)
         for(core = 0; core < p_used; core++) {

            int m_c_t, m_c_x;
            bool pad_m;

            if((m == Mb - 1) && m_pad) {
               m_c_t = (core == (p_l - 1) ? m_c1_last_core : m_c1);
               m_c_x = m_c1;
               pad_m = (core == (p_l - 1) ? 1 : 0);
            } else {
               m_c_t = m_c;
               m_c_x = m_c;
               pad_m = 0;
            }

            pack_ob_C_single_buf(&C[C_offset + core*m_c_x*N], &C_p[C_p_offset + core*m_c_x*n_c_t], 
               M, N, m1, n1, core*m_c_x, m_c_t, n_c_t, m_r, n_r, pad_m, pad_n);
         }

         C_p_offset += m_cb*n_c_t;
      }
   }
}




