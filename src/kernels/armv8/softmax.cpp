#include "kernels.h"
#include <stdio.h>
#include <arm_neon.h>
#include <math.h>
#include <cstdio>
#include <algorithm> // for std::min

/* -------- fast 4-lane exp  (stub – drop in your own) ----------------- */
static inline float32x4_t exp4_f32(float32x4_t x) {
    float tmp[4]; vst1q_f32(tmp, x);
    for (int i = 0; i < 4; ++i) tmp[i] = expf(tmp[i]);
    return vld1q_f32(tmp);
}

/* --------------------------------------------------------------------
 *  fused soft-max × V for an S tile of up to 8×12
 * ------------------------------------------------------------------*/
void softmax_matmul_fused_armv8_8x12(
        float       *S_tile,       /* actual_rows_S × actual_cols_S logits (row-major, with ld_S) */
        const float *V_panel,      /* actual_cols_S × d_v (row-major, with ld_V) */
        float       *A_panel,      /* actual_rows_S × d_v output block (row-major, with ld_A) */
        float       *row_max_vec,  /* actual_rows_S running maxima         */
        float       *row_den_vec,  /* actual_rows_S running denominators   */
        int          d_v,          /* value dimension, multiple of 4. ld_V and ld_A are also d_v here. */
        int          actual_rows_S,
        int          actual_cols_S,
        int          ld_S,         /* Leading dimension (pitch) of S_tile */
        int          ld_V,         /* Leading dimension (pitch) of V_panel */
        int          ld_A          /* Leading dimension (pitch) of A_panel */
        )
{
    for (int r = 0; r < actual_rows_S; ++r) {
        float *S_row = S_tile  + r * ld_S;
        float *A_row = A_panel + r * ld_A;

        float m_prev = row_max_vec[r];         // running max (may be -INF on first tile)
        float d_prev = row_den_vec[r];         // running denominator (0 on first tile)

        // At this point A_row already stores the running NORMALISED result for previous tiles
        // If first tile: A_row is zero-initialised and d_prev == 0.

        for (int c = 0; c < actual_cols_S; ++c) {
            float s_val = S_row[c];

            float m_new   = fmaxf(m_prev, s_val);
            float scale   = expf(m_prev - m_new);      // <= 1, well-defined even if m_prev=-INF (yields 0)
            scale        *= (m_prev > -INFINITY);      // zero if m_prev was -INF, else exp()

            float d_scaled = d_prev * scale;           // scale old denominator
            float s_exp    = expf(s_val - m_new);      // exponent for current column (unnormalised)
            float d_new    = d_scaled + s_exp;         // updated denominator

            // Compute blending factors to keep A_row NORMALISED with respect to d_new
            // old contribution weight
            float w_old = (d_new > 0.0f) ? (d_scaled / d_new) : 0.0f;   // safe when d_new==0
            float w_new = (d_new > 0.0f) ? (s_exp   / d_new) : 0.0f;

            float32x4_t w_old_vec = vdupq_n_f32(w_old);
            float32x4_t w_new_vec = vdupq_n_f32(w_new);
            const float *V_row = V_panel + c * ld_V;

            for (int k = 0; k < d_v; k += 4) {
                float32x4_t a_vec = vld1q_f32(A_row + k);      // previous normalised A
                float32x4_t v_vec = vld1q_f32(V_row + k);
                a_vec = vmulq_f32(a_vec, w_old_vec);           // scale old contribution
                a_vec = vfmaq_f32(a_vec, v_vec, w_new_vec);    // add new contribution (already normalised)
                vst1q_f32(A_row + k, a_vec);
            }

            // advance stats
            m_prev = m_new;
            d_prev = d_new;
        }

        // store running stats for this row (to be used by next tile)
        row_max_vec[r] = m_prev;
        row_den_vec[r] = d_prev;
    }
}
