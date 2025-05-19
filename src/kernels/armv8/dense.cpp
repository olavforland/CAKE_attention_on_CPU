

#include "kernels.h"

void cake_sgemm_armv8_8x12(float* A, float* B, float* C, int m, int n, int k) {
			  
	float32x4_t a, b1, b2, b3;
	float32x4_t c[8*3];
	
	// load tile of C into arm neon SIMD registers
	c[0]  = vld1q_f32(C);
	c[1]  = vld1q_f32(C + 4);
	c[2]  = vld1q_f32(C + 8);
	c[3]  = vld1q_f32(C + 12);
	c[4]  = vld1q_f32(C + 16);
	c[5]  = vld1q_f32(C + 20);
	c[6]  = vld1q_f32(C + 24);
	c[7]  = vld1q_f32(C + 28);
	c[8]  = vld1q_f32(C + 32);
	c[9]  = vld1q_f32(C + 36);
	c[10]  = vld1q_f32(C + 40);
	c[11]  = vld1q_f32(C + 44);
	c[12]  = vld1q_f32(C + 48);
	c[13]  = vld1q_f32(C + 52);
	c[14]  = vld1q_f32(C + 56);
	c[15]  = vld1q_f32(C + 60);
	c[16]  = vld1q_f32(C + 64);
	c[17]  = vld1q_f32(C + 68);
	c[18]  = vld1q_f32(C + 72);
	c[19]  = vld1q_f32(C + 76);
	c[20]  = vld1q_f32(C + 80);
	c[21]  = vld1q_f32(C + 84);
	c[22]  = vld1q_f32(C + 88);
	c[23]  = vld1q_f32(C + 92);

	int rem = k % 4;
	k -= rem;

	// outer-product unrolled 4 times
	for(int kk = 0; kk < k; kk += 4) { 

			
		b1 = vld1q_f32(B);
		b2 = vld1q_f32(B + 4);
		b3 = vld1q_f32(B + 8);

		a = vld1q_dup_f32(A++);
		c[0] =  vfmaq_f32(c[0], b1, a);
		c[1] =  vfmaq_f32(c[1], b2, a);
		c[2] =  vfmaq_f32(c[2], b3, a);

		a = vld1q_dup_f32(A++);
		c[3] =  vfmaq_f32(c[3], b1, a);
		c[4] =  vfmaq_f32(c[4], b2, a);
		c[5] =  vfmaq_f32(c[5], b3, a);

		a = vld1q_dup_f32(A++);
		c[6] =  vfmaq_f32(c[6], b1, a);
		c[7] =  vfmaq_f32(c[7], b2, a);
		c[8] =  vfmaq_f32(c[8], b3, a);

		a = vld1q_dup_f32(A++);
		c[9] =  vfmaq_f32(c[9], b1, a);
		c[10] =  vfmaq_f32(c[10], b2, a);
		c[11] =  vfmaq_f32(c[11], b3, a);

		a = vld1q_dup_f32(A++);
		c[12] =  vfmaq_f32(c[12], b1, a);
		c[13] =  vfmaq_f32(c[13], b2, a);
		c[14] =  vfmaq_f32(c[14], b3, a);

		a = vld1q_dup_f32(A++);
		c[15] =  vfmaq_f32(c[15], b1, a);
		c[16] =  vfmaq_f32(c[16], b2, a);
		c[17] =  vfmaq_f32(c[17], b3, a);

		a = vld1q_dup_f32(A++);
		c[18] =  vfmaq_f32(c[18], b1, a);
		c[19] =  vfmaq_f32(c[19], b2, a);
		c[20] =  vfmaq_f32(c[20], b3, a);

		a = vld1q_dup_f32(A++);
		c[21] =  vfmaq_f32(c[21], b1, a);
		c[22] =  vfmaq_f32(c[22], b2, a);
		c[23] =  vfmaq_f32(c[23], b3, a);

		B += n;


		b1 = vld1q_f32(B);
		b2 = vld1q_f32(B + 4);
		b3 = vld1q_f32(B + 8);

		a = vld1q_dup_f32(A++);
		c[0] =  vfmaq_f32(c[0], b1, a);
		c[1] =  vfmaq_f32(c[1], b2, a);
		c[2] =  vfmaq_f32(c[2], b3, a);

		a = vld1q_dup_f32(A++);
		c[3] =  vfmaq_f32(c[3], b1, a);
		c[4] =  vfmaq_f32(c[4], b2, a);
		c[5] =  vfmaq_f32(c[5], b3, a);

		a = vld1q_dup_f32(A++);
		c[6] =  vfmaq_f32(c[6], b1, a);
		c[7] =  vfmaq_f32(c[7], b2, a);
		c[8] =  vfmaq_f32(c[8], b3, a);

		a = vld1q_dup_f32(A++);
		c[9] =  vfmaq_f32(c[9], b1, a);
		c[10] =  vfmaq_f32(c[10], b2, a);
		c[11] =  vfmaq_f32(c[11], b3, a);

		a = vld1q_dup_f32(A++);
		c[12] =  vfmaq_f32(c[12], b1, a);
		c[13] =  vfmaq_f32(c[13], b2, a);
		c[14] =  vfmaq_f32(c[14], b3, a);

		a = vld1q_dup_f32(A++);
		c[15] =  vfmaq_f32(c[15], b1, a);
		c[16] =  vfmaq_f32(c[16], b2, a);
		c[17] =  vfmaq_f32(c[17], b3, a);

		a = vld1q_dup_f32(A++);
		c[18] =  vfmaq_f32(c[18], b1, a);
		c[19] =  vfmaq_f32(c[19], b2, a);
		c[20] =  vfmaq_f32(c[20], b3, a);

		a = vld1q_dup_f32(A++);
		c[21] =  vfmaq_f32(c[21], b1, a);
		c[22] =  vfmaq_f32(c[22], b2, a);
		c[23] =  vfmaq_f32(c[23], b3, a);

		B += n;


		b1 = vld1q_f32(B);
		b2 = vld1q_f32(B + 4);
		b3 = vld1q_f32(B + 8);

		a = vld1q_dup_f32(A++);
		c[0] =  vfmaq_f32(c[0], b1, a);
		c[1] =  vfmaq_f32(c[1], b2, a);
		c[2] =  vfmaq_f32(c[2], b3, a);

		a = vld1q_dup_f32(A++);
		c[3] =  vfmaq_f32(c[3], b1, a);
		c[4] =  vfmaq_f32(c[4], b2, a);
		c[5] =  vfmaq_f32(c[5], b3, a);

		a = vld1q_dup_f32(A++);
		c[6] =  vfmaq_f32(c[6], b1, a);
		c[7] =  vfmaq_f32(c[7], b2, a);
		c[8] =  vfmaq_f32(c[8], b3, a);

		a = vld1q_dup_f32(A++);
		c[9] =  vfmaq_f32(c[9], b1, a);
		c[10] =  vfmaq_f32(c[10], b2, a);
		c[11] =  vfmaq_f32(c[11], b3, a);

		a = vld1q_dup_f32(A++);
		c[12] =  vfmaq_f32(c[12], b1, a);
		c[13] =  vfmaq_f32(c[13], b2, a);
		c[14] =  vfmaq_f32(c[14], b3, a);

		a = vld1q_dup_f32(A++);
		c[15] =  vfmaq_f32(c[15], b1, a);
		c[16] =  vfmaq_f32(c[16], b2, a);
		c[17] =  vfmaq_f32(c[17], b3, a);

		a = vld1q_dup_f32(A++);
		c[18] =  vfmaq_f32(c[18], b1, a);
		c[19] =  vfmaq_f32(c[19], b2, a);
		c[20] =  vfmaq_f32(c[20], b3, a);

		a = vld1q_dup_f32(A++);
		c[21] =  vfmaq_f32(c[21], b1, a);
		c[22] =  vfmaq_f32(c[22], b2, a);
		c[23] =  vfmaq_f32(c[23], b3, a);

		B += n;


		b1 = vld1q_f32(B);
		b2 = vld1q_f32(B + 4);
		b3 = vld1q_f32(B + 8);

		a = vld1q_dup_f32(A++);
		c[0] =  vfmaq_f32(c[0], b1, a);
		c[1] =  vfmaq_f32(c[1], b2, a);
		c[2] =  vfmaq_f32(c[2], b3, a);

		a = vld1q_dup_f32(A++);
		c[3] =  vfmaq_f32(c[3], b1, a);
		c[4] =  vfmaq_f32(c[4], b2, a);
		c[5] =  vfmaq_f32(c[5], b3, a);

		a = vld1q_dup_f32(A++);
		c[6] =  vfmaq_f32(c[6], b1, a);
		c[7] =  vfmaq_f32(c[7], b2, a);
		c[8] =  vfmaq_f32(c[8], b3, a);

		a = vld1q_dup_f32(A++);
		c[9] =  vfmaq_f32(c[9], b1, a);
		c[10] =  vfmaq_f32(c[10], b2, a);
		c[11] =  vfmaq_f32(c[11], b3, a);

		a = vld1q_dup_f32(A++);
		c[12] =  vfmaq_f32(c[12], b1, a);
		c[13] =  vfmaq_f32(c[13], b2, a);
		c[14] =  vfmaq_f32(c[14], b3, a);

		a = vld1q_dup_f32(A++);
		c[15] =  vfmaq_f32(c[15], b1, a);
		c[16] =  vfmaq_f32(c[16], b2, a);
		c[17] =  vfmaq_f32(c[17], b3, a);

		a = vld1q_dup_f32(A++);
		c[18] =  vfmaq_f32(c[18], b1, a);
		c[19] =  vfmaq_f32(c[19], b2, a);
		c[20] =  vfmaq_f32(c[20], b3, a);

		a = vld1q_dup_f32(A++);
		c[21] =  vfmaq_f32(c[21], b1, a);
		c[22] =  vfmaq_f32(c[22], b2, a);
		c[23] =  vfmaq_f32(c[23], b3, a);

		B += n;
	}
			
	for(int kk = 0; kk < rem; kk++) { 
			
		b1 = vld1q_f32(B);
		b2 = vld1q_f32(B + 4);
		b3 = vld1q_f32(B + 8);

		a = vld1q_dup_f32(A++);
		c[0] =  vfmaq_f32(c[0], b1, a);
		c[1] =  vfmaq_f32(c[1], b2, a);
		c[2] =  vfmaq_f32(c[2], b3, a);

		a = vld1q_dup_f32(A++);
		c[3] =  vfmaq_f32(c[3], b1, a);
		c[4] =  vfmaq_f32(c[4], b2, a);
		c[5] =  vfmaq_f32(c[5], b3, a);

		a = vld1q_dup_f32(A++);
		c[6] =  vfmaq_f32(c[6], b1, a);
		c[7] =  vfmaq_f32(c[7], b2, a);
		c[8] =  vfmaq_f32(c[8], b3, a);

		a = vld1q_dup_f32(A++);
		c[9] =  vfmaq_f32(c[9], b1, a);
		c[10] =  vfmaq_f32(c[10], b2, a);
		c[11] =  vfmaq_f32(c[11], b3, a);

		a = vld1q_dup_f32(A++);
		c[12] =  vfmaq_f32(c[12], b1, a);
		c[13] =  vfmaq_f32(c[13], b2, a);
		c[14] =  vfmaq_f32(c[14], b3, a);

		a = vld1q_dup_f32(A++);
		c[15] =  vfmaq_f32(c[15], b1, a);
		c[16] =  vfmaq_f32(c[16], b2, a);
		c[17] =  vfmaq_f32(c[17], b3, a);

		a = vld1q_dup_f32(A++);
		c[18] =  vfmaq_f32(c[18], b1, a);
		c[19] =  vfmaq_f32(c[19], b2, a);
		c[20] =  vfmaq_f32(c[20], b3, a);

		a = vld1q_dup_f32(A++);
		c[21] =  vfmaq_f32(c[21], b1, a);
		c[22] =  vfmaq_f32(c[22], b2, a);
		c[23] =  vfmaq_f32(c[23], b3, a);

		B += n;
			}
			
	vst1q_f32(C, c[0]);
	vst1q_f32(C + 4, c[1]);
	vst1q_f32(C + 8, c[2]);
	vst1q_f32(C + 12, c[3]);
	vst1q_f32(C + 16, c[4]);
	vst1q_f32(C + 20, c[5]);
	vst1q_f32(C + 24, c[6]);
	vst1q_f32(C + 28, c[7]);
	vst1q_f32(C + 32, c[8]);
	vst1q_f32(C + 36, c[9]);
	vst1q_f32(C + 40, c[10]);
	vst1q_f32(C + 44, c[11]);
	vst1q_f32(C + 48, c[12]);
	vst1q_f32(C + 52, c[13]);
	vst1q_f32(C + 56, c[14]);
	vst1q_f32(C + 60, c[15]);
	vst1q_f32(C + 64, c[16]);
	vst1q_f32(C + 68, c[17]);
	vst1q_f32(C + 72, c[18]);
	vst1q_f32(C + 76, c[19]);
	vst1q_f32(C + 80, c[20]);
	vst1q_f32(C + 84, c[21]);
	vst1q_f32(C + 88, c[22]);
	vst1q_f32(C + 92, c[23]);
}

void cake_sgemm_armv8_8x24(float* A, float* B, float* C, int m, int n, int k) {
			  
	float32x4_t a, b1, b2, b3, b4, b5, b6;
	float32x4_t c[8*6];
	
	// load tile of C into arm neon SIMD registers
	c[0]  = vld1q_f32(C);
	c[1]  = vld1q_f32(C + 4);
	c[2]  = vld1q_f32(C + 8);
	c[3]  = vld1q_f32(C + 12);
	c[4]  = vld1q_f32(C + 16);
	c[5]  = vld1q_f32(C + 20);
	c[6]  = vld1q_f32(C + 24);
	c[7]  = vld1q_f32(C + 28);
	c[8]  = vld1q_f32(C + 32);
	c[9]  = vld1q_f32(C + 36);
	c[10]  = vld1q_f32(C + 40);
	c[11]  = vld1q_f32(C + 44);
	c[12]  = vld1q_f32(C + 48);
	c[13]  = vld1q_f32(C + 52);
	c[14]  = vld1q_f32(C + 56);
	c[15]  = vld1q_f32(C + 60);
	c[16]  = vld1q_f32(C + 64);
	c[17]  = vld1q_f32(C + 68);
	c[18]  = vld1q_f32(C + 72);
	c[19]  = vld1q_f32(C + 76);
	c[20]  = vld1q_f32(C + 80);
	c[21]  = vld1q_f32(C + 84);
	c[22]  = vld1q_f32(C + 88);
	c[23]  = vld1q_f32(C + 92);
	c[24]  = vld1q_f32(C + 96);
	c[25]  = vld1q_f32(C + 100);
	c[26]  = vld1q_f32(C + 104);
	c[27]  = vld1q_f32(C + 108);
	c[28]  = vld1q_f32(C + 112);
	c[29]  = vld1q_f32(C + 116);
	c[30]  = vld1q_f32(C + 120);
	c[31]  = vld1q_f32(C + 124);
	c[32]  = vld1q_f32(C + 128);
	c[33]  = vld1q_f32(C + 132);
	c[34]  = vld1q_f32(C + 136);
	c[35]  = vld1q_f32(C + 140);
	c[36]  = vld1q_f32(C + 144);
	c[37]  = vld1q_f32(C + 148);
	c[38]  = vld1q_f32(C + 152);
	c[39]  = vld1q_f32(C + 156);
	c[40]  = vld1q_f32(C + 160);
	c[41]  = vld1q_f32(C + 164);
	c[42]  = vld1q_f32(C + 168);
	c[43]  = vld1q_f32(C + 172);
	c[44]  = vld1q_f32(C + 176);
	c[45]  = vld1q_f32(C + 180);
	c[46]  = vld1q_f32(C + 184);
	c[47]  = vld1q_f32(C + 188);

	int rem = k % 4;
	k -= rem;

	// outer-product unrolled 4 times
	for(int kk = 0; kk < k; kk += 4) { 

			
		b1 = vld1q_f32(B);
		b2 = vld1q_f32(B + 4);
		b3 = vld1q_f32(B + 8);
		b4 = vld1q_f32(B + 12);
		b5 = vld1q_f32(B + 16);
		b6 = vld1q_f32(B + 20);

		a = vld1q_dup_f32(A++);
		c[0] =  vfmaq_f32(c[0], b1, a);
		c[1] =  vfmaq_f32(c[1], b2, a);
		c[2] =  vfmaq_f32(c[2], b3, a);
		c[3] =  vfmaq_f32(c[3], b4, a);
		c[4] =  vfmaq_f32(c[4], b5, a);
		c[5] =  vfmaq_f32(c[5], b6, a);

		a = vld1q_dup_f32(A++);
		c[6] =  vfmaq_f32(c[6], b1, a);
		c[7] =  vfmaq_f32(c[7], b2, a);
		c[8] =  vfmaq_f32(c[8], b3, a);
		c[9] =  vfmaq_f32(c[9], b4, a);
		c[10] =  vfmaq_f32(c[10], b5, a);
		c[11] =  vfmaq_f32(c[11], b6, a);

		a = vld1q_dup_f32(A++);
		c[12] =  vfmaq_f32(c[12], b1, a);
		c[13] =  vfmaq_f32(c[13], b2, a);
		c[14] =  vfmaq_f32(c[14], b3, a);
		c[15] =  vfmaq_f32(c[15], b4, a);
		c[16] =  vfmaq_f32(c[16], b5, a);
		c[17] =  vfmaq_f32(c[17], b6, a);

		a = vld1q_dup_f32(A++);
		c[18] =  vfmaq_f32(c[18], b1, a);
		c[19] =  vfmaq_f32(c[19], b2, a);
		c[20] =  vfmaq_f32(c[20], b3, a);
		c[21] =  vfmaq_f32(c[21], b4, a);
		c[22] =  vfmaq_f32(c[22], b5, a);
		c[23] =  vfmaq_f32(c[23], b6, a);

		a = vld1q_dup_f32(A++);
		c[24] =  vfmaq_f32(c[24], b1, a);
		c[25] =  vfmaq_f32(c[25], b2, a);
		c[26] =  vfmaq_f32(c[26], b3, a);
		c[27] =  vfmaq_f32(c[27], b4, a);
		c[28] =  vfmaq_f32(c[28], b5, a);
		c[29] =  vfmaq_f32(c[29], b6, a);

		a = vld1q_dup_f32(A++);
		c[30] =  vfmaq_f32(c[30], b1, a);
		c[31] =  vfmaq_f32(c[31], b2, a);
		c[32] =  vfmaq_f32(c[32], b3, a);
		c[33] =  vfmaq_f32(c[33], b4, a);
		c[34] =  vfmaq_f32(c[34], b5, a);
		c[35] =  vfmaq_f32(c[35], b6, a);

		a = vld1q_dup_f32(A++);
		c[36] =  vfmaq_f32(c[36], b1, a);
		c[37] =  vfmaq_f32(c[37], b2, a);
		c[38] =  vfmaq_f32(c[38], b3, a);
		c[39] =  vfmaq_f32(c[39], b4, a);
		c[40] =  vfmaq_f32(c[40], b5, a);
		c[41] =  vfmaq_f32(c[41], b6, a);

		a = vld1q_dup_f32(A++);
		c[42] =  vfmaq_f32(c[42], b1, a);
		c[43] =  vfmaq_f32(c[43], b2, a);
		c[44] =  vfmaq_f32(c[44], b3, a);
		c[45] =  vfmaq_f32(c[45], b4, a);
		c[46] =  vfmaq_f32(c[46], b5, a);
		c[47] =  vfmaq_f32(c[47], b6, a);

		B += n;


		b1 = vld1q_f32(B);
		b2 = vld1q_f32(B + 4);
		b3 = vld1q_f32(B + 8);
		b4 = vld1q_f32(B + 12);
		b5 = vld1q_f32(B + 16);
		b6 = vld1q_f32(B + 20);

		a = vld1q_dup_f32(A++);
		c[0] =  vfmaq_f32(c[0], b1, a);
		c[1] =  vfmaq_f32(c[1], b2, a);
		c[2] =  vfmaq_f32(c[2], b3, a);
		c[3] =  vfmaq_f32(c[3], b4, a);
		c[4] =  vfmaq_f32(c[4], b5, a);
		c[5] =  vfmaq_f32(c[5], b6, a);

		a = vld1q_dup_f32(A++);
		c[6] =  vfmaq_f32(c[6], b1, a);
		c[7] =  vfmaq_f32(c[7], b2, a);
		c[8] =  vfmaq_f32(c[8], b3, a);
		c[9] =  vfmaq_f32(c[9], b4, a);
		c[10] =  vfmaq_f32(c[10], b5, a);
		c[11] =  vfmaq_f32(c[11], b6, a);

		a = vld1q_dup_f32(A++);
		c[12] =  vfmaq_f32(c[12], b1, a);
		c[13] =  vfmaq_f32(c[13], b2, a);
		c[14] =  vfmaq_f32(c[14], b3, a);
		c[15] =  vfmaq_f32(c[15], b4, a);
		c[16] =  vfmaq_f32(c[16], b5, a);
		c[17] =  vfmaq_f32(c[17], b6, a);

		a = vld1q_dup_f32(A++);
		c[18] =  vfmaq_f32(c[18], b1, a);
		c[19] =  vfmaq_f32(c[19], b2, a);
		c[20] =  vfmaq_f32(c[20], b3, a);
		c[21] =  vfmaq_f32(c[21], b4, a);
		c[22] =  vfmaq_f32(c[22], b5, a);
		c[23] =  vfmaq_f32(c[23], b6, a);

		a = vld1q_dup_f32(A++);
		c[24] =  vfmaq_f32(c[24], b1, a);
		c[25] =  vfmaq_f32(c[25], b2, a);
		c[26] =  vfmaq_f32(c[26], b3, a);
		c[27] =  vfmaq_f32(c[27], b4, a);
		c[28] =  vfmaq_f32(c[28], b5, a);
		c[29] =  vfmaq_f32(c[29], b6, a);

		a = vld1q_dup_f32(A++);
		c[30] =  vfmaq_f32(c[30], b1, a);
		c[31] =  vfmaq_f32(c[31], b2, a);
		c[32] =  vfmaq_f32(c[32], b3, a);
		c[33] =  vfmaq_f32(c[33], b4, a);
		c[34] =  vfmaq_f32(c[34], b5, a);
		c[35] =  vfmaq_f32(c[35], b6, a);

		a = vld1q_dup_f32(A++);
		c[36] =  vfmaq_f32(c[36], b1, a);
		c[37] =  vfmaq_f32(c[37], b2, a);
		c[38] =  vfmaq_f32(c[38], b3, a);
		c[39] =  vfmaq_f32(c[39], b4, a);
		c[40] =  vfmaq_f32(c[40], b5, a);
		c[41] =  vfmaq_f32(c[41], b6, a);

		a = vld1q_dup_f32(A++);
		c[42] =  vfmaq_f32(c[42], b1, a);
		c[43] =  vfmaq_f32(c[43], b2, a);
		c[44] =  vfmaq_f32(c[44], b3, a);
		c[45] =  vfmaq_f32(c[45], b4, a);
		c[46] =  vfmaq_f32(c[46], b5, a);
		c[47] =  vfmaq_f32(c[47], b6, a);

		B += n;


		b1 = vld1q_f32(B);
		b2 = vld1q_f32(B + 4);
		b3 = vld1q_f32(B + 8);
		b4 = vld1q_f32(B + 12);
		b5 = vld1q_f32(B + 16);
		b6 = vld1q_f32(B + 20);

		a = vld1q_dup_f32(A++);
		c[0] =  vfmaq_f32(c[0], b1, a);
		c[1] =  vfmaq_f32(c[1], b2, a);
		c[2] =  vfmaq_f32(c[2], b3, a);
		c[3] =  vfmaq_f32(c[3], b4, a);
		c[4] =  vfmaq_f32(c[4], b5, a);
		c[5] =  vfmaq_f32(c[5], b6, a);

		a = vld1q_dup_f32(A++);
		c[6] =  vfmaq_f32(c[6], b1, a);
		c[7] =  vfmaq_f32(c[7], b2, a);
		c[8] =  vfmaq_f32(c[8], b3, a);
		c[9] =  vfmaq_f32(c[9], b4, a);
		c[10] =  vfmaq_f32(c[10], b5, a);
		c[11] =  vfmaq_f32(c[11], b6, a);

		a = vld1q_dup_f32(A++);
		c[12] =  vfmaq_f32(c[12], b1, a);
		c[13] =  vfmaq_f32(c[13], b2, a);
		c[14] =  vfmaq_f32(c[14], b3, a);
		c[15] =  vfmaq_f32(c[15], b4, a);
		c[16] =  vfmaq_f32(c[16], b5, a);
		c[17] =  vfmaq_f32(c[17], b6, a);

		a = vld1q_dup_f32(A++);
		c[18] =  vfmaq_f32(c[18], b1, a);
		c[19] =  vfmaq_f32(c[19], b2, a);
		c[20] =  vfmaq_f32(c[20], b3, a);
		c[21] =  vfmaq_f32(c[21], b4, a);
		c[22] =  vfmaq_f32(c[22], b5, a);
		c[23] =  vfmaq_f32(c[23], b6, a);

		a = vld1q_dup_f32(A++);
		c[24] =  vfmaq_f32(c[24], b1, a);
		c[25] =  vfmaq_f32(c[25], b2, a);
		c[26] =  vfmaq_f32(c[26], b3, a);
		c[27] =  vfmaq_f32(c[27], b4, a);
		c[28] =  vfmaq_f32(c[28], b5, a);
		c[29] =  vfmaq_f32(c[29], b6, a);

		a = vld1q_dup_f32(A++);
		c[30] =  vfmaq_f32(c[30], b1, a);
		c[31] =  vfmaq_f32(c[31], b2, a);
		c[32] =  vfmaq_f32(c[32], b3, a);
		c[33] =  vfmaq_f32(c[33], b4, a);
		c[34] =  vfmaq_f32(c[34], b5, a);
		c[35] =  vfmaq_f32(c[35], b6, a);

		a = vld1q_dup_f32(A++);
		c[36] =  vfmaq_f32(c[36], b1, a);
		c[37] =  vfmaq_f32(c[37], b2, a);
		c[38] =  vfmaq_f32(c[38], b3, a);
		c[39] =  vfmaq_f32(c[39], b4, a);
		c[40] =  vfmaq_f32(c[40], b5, a);
		c[41] =  vfmaq_f32(c[41], b6, a);

		a = vld1q_dup_f32(A++);
		c[42] =  vfmaq_f32(c[42], b1, a);
		c[43] =  vfmaq_f32(c[43], b2, a);
		c[44] =  vfmaq_f32(c[44], b3, a);
		c[45] =  vfmaq_f32(c[45], b4, a);
		c[46] =  vfmaq_f32(c[46], b5, a);
		c[47] =  vfmaq_f32(c[47], b6, a);

		B += n;


		b1 = vld1q_f32(B);
		b2 = vld1q_f32(B + 4);
		b3 = vld1q_f32(B + 8);
		b4 = vld1q_f32(B + 12);
		b5 = vld1q_f32(B + 16);
		b6 = vld1q_f32(B + 20);

		a = vld1q_dup_f32(A++);
		c[0] =  vfmaq_f32(c[0], b1, a);
		c[1] =  vfmaq_f32(c[1], b2, a);
		c[2] =  vfmaq_f32(c[2], b3, a);
		c[3] =  vfmaq_f32(c[3], b4, a);
		c[4] =  vfmaq_f32(c[4], b5, a);
		c[5] =  vfmaq_f32(c[5], b6, a);

		a = vld1q_dup_f32(A++);
		c[6] =  vfmaq_f32(c[6], b1, a);
		c[7] =  vfmaq_f32(c[7], b2, a);
		c[8] =  vfmaq_f32(c[8], b3, a);
		c[9] =  vfmaq_f32(c[9], b4, a);
		c[10] =  vfmaq_f32(c[10], b5, a);
		c[11] =  vfmaq_f32(c[11], b6, a);

		a = vld1q_dup_f32(A++);
		c[12] =  vfmaq_f32(c[12], b1, a);
		c[13] =  vfmaq_f32(c[13], b2, a);
		c[14] =  vfmaq_f32(c[14], b3, a);
		c[15] =  vfmaq_f32(c[15], b4, a);
		c[16] =  vfmaq_f32(c[16], b5, a);
		c[17] =  vfmaq_f32(c[17], b6, a);

		a = vld1q_dup_f32(A++);
		c[18] =  vfmaq_f32(c[18], b1, a);
		c[19] =  vfmaq_f32(c[19], b2, a);
		c[20] =  vfmaq_f32(c[20], b3, a);
		c[21] =  vfmaq_f32(c[21], b4, a);
		c[22] =  vfmaq_f32(c[22], b5, a);
		c[23] =  vfmaq_f32(c[23], b6, a);

		a = vld1q_dup_f32(A++);
		c[24] =  vfmaq_f32(c[24], b1, a);
		c[25] =  vfmaq_f32(c[25], b2, a);
		c[26] =  vfmaq_f32(c[26], b3, a);
		c[27] =  vfmaq_f32(c[27], b4, a);
		c[28] =  vfmaq_f32(c[28], b5, a);
		c[29] =  vfmaq_f32(c[29], b6, a);

		a = vld1q_dup_f32(A++);
		c[30] =  vfmaq_f32(c[30], b1, a);
		c[31] =  vfmaq_f32(c[31], b2, a);
		c[32] =  vfmaq_f32(c[32], b3, a);
		c[33] =  vfmaq_f32(c[33], b4, a);
		c[34] =  vfmaq_f32(c[34], b5, a);
		c[35] =  vfmaq_f32(c[35], b6, a);

		a = vld1q_dup_f32(A++);
		c[36] =  vfmaq_f32(c[36], b1, a);
		c[37] =  vfmaq_f32(c[37], b2, a);
		c[38] =  vfmaq_f32(c[38], b3, a);
		c[39] =  vfmaq_f32(c[39], b4, a);
		c[40] =  vfmaq_f32(c[40], b5, a);
		c[41] =  vfmaq_f32(c[41], b6, a);

		a = vld1q_dup_f32(A++);
		c[42] =  vfmaq_f32(c[42], b1, a);
		c[43] =  vfmaq_f32(c[43], b2, a);
		c[44] =  vfmaq_f32(c[44], b3, a);
		c[45] =  vfmaq_f32(c[45], b4, a);
		c[46] =  vfmaq_f32(c[46], b5, a);
		c[47] =  vfmaq_f32(c[47], b6, a);

		B += n;


	}
			
	for(int kk = 0; kk < rem; kk++) { 
			
		b1 = vld1q_f32(B);
		b2 = vld1q_f32(B + 4);
		b3 = vld1q_f32(B + 8);
		b4 = vld1q_f32(B + 12);
		b5 = vld1q_f32(B + 16);
		b6 = vld1q_f32(B + 20);

		a = vld1q_dup_f32(A++);
		c[0] =  vfmaq_f32(c[0], b1, a);
		c[1] =  vfmaq_f32(c[1], b2, a);
		c[2] =  vfmaq_f32(c[2], b3, a);
		c[3] =  vfmaq_f32(c[3], b4, a);
		c[4] =  vfmaq_f32(c[4], b5, a);
		c[5] =  vfmaq_f32(c[5], b6, a);

		a = vld1q_dup_f32(A++);
		c[6] =  vfmaq_f32(c[6], b1, a);
		c[7] =  vfmaq_f32(c[7], b2, a);
		c[8] =  vfmaq_f32(c[8], b3, a);
		c[9] =  vfmaq_f32(c[9], b4, a);
		c[10] =  vfmaq_f32(c[10], b5, a);
		c[11] =  vfmaq_f32(c[11], b6, a);

		a = vld1q_dup_f32(A++);
		c[12] =  vfmaq_f32(c[12], b1, a);
		c[13] =  vfmaq_f32(c[13], b2, a);
		c[14] =  vfmaq_f32(c[14], b3, a);
		c[15] =  vfmaq_f32(c[15], b4, a);
		c[16] =  vfmaq_f32(c[16], b5, a);
		c[17] =  vfmaq_f32(c[17], b6, a);

		a = vld1q_dup_f32(A++);
		c[18] =  vfmaq_f32(c[18], b1, a);
		c[19] =  vfmaq_f32(c[19], b2, a);
		c[20] =  vfmaq_f32(c[20], b3, a);
		c[21] =  vfmaq_f32(c[21], b4, a);
		c[22] =  vfmaq_f32(c[22], b5, a);
		c[23] =  vfmaq_f32(c[23], b6, a);

		a = vld1q_dup_f32(A++);
		c[24] =  vfmaq_f32(c[24], b1, a);
		c[25] =  vfmaq_f32(c[25], b2, a);
		c[26] =  vfmaq_f32(c[26], b3, a);
		c[27] =  vfmaq_f32(c[27], b4, a);
		c[28] =  vfmaq_f32(c[28], b5, a);
		c[29] =  vfmaq_f32(c[29], b6, a);

		a = vld1q_dup_f32(A++);
		c[30] =  vfmaq_f32(c[30], b1, a);
		c[31] =  vfmaq_f32(c[31], b2, a);
		c[32] =  vfmaq_f32(c[32], b3, a);
		c[33] =  vfmaq_f32(c[33], b4, a);
		c[34] =  vfmaq_f32(c[34], b5, a);
		c[35] =  vfmaq_f32(c[35], b6, a);

		a = vld1q_dup_f32(A++);
		c[36] =  vfmaq_f32(c[36], b1, a);
		c[37] =  vfmaq_f32(c[37], b2, a);
		c[38] =  vfmaq_f32(c[38], b3, a);
		c[39] =  vfmaq_f32(c[39], b4, a);
		c[40] =  vfmaq_f32(c[40], b5, a);
		c[41] =  vfmaq_f32(c[41], b6, a);

		a = vld1q_dup_f32(A++);
		c[42] =  vfmaq_f32(c[42], b1, a);
		c[43] =  vfmaq_f32(c[43], b2, a);
		c[44] =  vfmaq_f32(c[44], b3, a);
		c[45] =  vfmaq_f32(c[45], b4, a);
		c[46] =  vfmaq_f32(c[46], b5, a);
		c[47] =  vfmaq_f32(c[47], b6, a);

		B += n;
			}
			
	vst1q_f32(C, c[0]);
	vst1q_f32(C + 4, c[1]);
	vst1q_f32(C + 8, c[2]);
	vst1q_f32(C + 12, c[3]);
	vst1q_f32(C + 16, c[4]);
	vst1q_f32(C + 20, c[5]);
	vst1q_f32(C + 24, c[6]);
	vst1q_f32(C + 28, c[7]);
	vst1q_f32(C + 32, c[8]);
	vst1q_f32(C + 36, c[9]);
	vst1q_f32(C + 40, c[10]);
	vst1q_f32(C + 44, c[11]);
	vst1q_f32(C + 48, c[12]);
	vst1q_f32(C + 52, c[13]);
	vst1q_f32(C + 56, c[14]);
	vst1q_f32(C + 60, c[15]);
	vst1q_f32(C + 64, c[16]);
	vst1q_f32(C + 68, c[17]);
	vst1q_f32(C + 72, c[18]);
	vst1q_f32(C + 76, c[19]);
	vst1q_f32(C + 80, c[20]);
	vst1q_f32(C + 84, c[21]);
	vst1q_f32(C + 88, c[22]);
	vst1q_f32(C + 92, c[23]);
	vst1q_f32(C + 96, c[24]);
	vst1q_f32(C + 100, c[25]);
	vst1q_f32(C + 104, c[26]);
	vst1q_f32(C + 108, c[27]);
	vst1q_f32(C + 112, c[28]);
	vst1q_f32(C + 116, c[29]);
	vst1q_f32(C + 120, c[30]);
	vst1q_f32(C + 124, c[31]);
	vst1q_f32(C + 128, c[32]);
	vst1q_f32(C + 132, c[33]);
	vst1q_f32(C + 136, c[34]);
	vst1q_f32(C + 140, c[35]);
	vst1q_f32(C + 144, c[36]);
	vst1q_f32(C + 148, c[37]);
	vst1q_f32(C + 152, c[38]);
	vst1q_f32(C + 156, c[39]);
	vst1q_f32(C + 160, c[40]);
	vst1q_f32(C + 164, c[41]);
	vst1q_f32(C + 168, c[42]);
	vst1q_f32(C + 172, c[43]);
	vst1q_f32(C + 176, c[44]);
	vst1q_f32(C + 180, c[45]);
	vst1q_f32(C + 184, c[46]);
	vst1q_f32(C + 188, c[47]);
}
void cake_sgemm_armv8_10x12(float* A, float* B, float* C, int m, int n, int k) {
			  
	float32x4_t a, b1, b2, b3;
	float32x4_t c[10*3];
	
	// load tile of C into arm neon SIMD registers
	c[0]  = vld1q_f32(C);
	c[1]  = vld1q_f32(C + 4);
	c[2]  = vld1q_f32(C + 8);
	c[3]  = vld1q_f32(C + 12);
	c[4]  = vld1q_f32(C + 16);
	c[5]  = vld1q_f32(C + 20);
	c[6]  = vld1q_f32(C + 24);
	c[7]  = vld1q_f32(C + 28);
	c[8]  = vld1q_f32(C + 32);
	c[9]  = vld1q_f32(C + 36);
	c[10]  = vld1q_f32(C + 40);
	c[11]  = vld1q_f32(C + 44);
	c[12]  = vld1q_f32(C + 48);
	c[13]  = vld1q_f32(C + 52);
	c[14]  = vld1q_f32(C + 56);
	c[15]  = vld1q_f32(C + 60);
	c[16]  = vld1q_f32(C + 64);
	c[17]  = vld1q_f32(C + 68);
	c[18]  = vld1q_f32(C + 72);
	c[19]  = vld1q_f32(C + 76);
	c[20]  = vld1q_f32(C + 80);
	c[21]  = vld1q_f32(C + 84);
	c[22]  = vld1q_f32(C + 88);
	c[23]  = vld1q_f32(C + 92);
	c[24]  = vld1q_f32(C + 96);
	c[25]  = vld1q_f32(C + 100);
	c[26]  = vld1q_f32(C + 104);
	c[27]  = vld1q_f32(C + 108);
	c[28]  = vld1q_f32(C + 112);
	c[29]  = vld1q_f32(C + 116);

	int rem = k % 4;
	k -= rem;

	// outer-product unrolled 4 times
	for(int kk = 0; kk < k; kk += 4) { 

			
		b1 = vld1q_f32(B);
		b2 = vld1q_f32(B + 4);
		b3 = vld1q_f32(B + 8);

		a = vld1q_dup_f32(A++);
		c[0] =  vfmaq_f32(c[0], b1, a);
		c[1] =  vfmaq_f32(c[1], b2, a);
		c[2] =  vfmaq_f32(c[2], b3, a);

		a = vld1q_dup_f32(A++);
		c[3] =  vfmaq_f32(c[3], b1, a);
		c[4] =  vfmaq_f32(c[4], b2, a);
		c[5] =  vfmaq_f32(c[5], b3, a);

		a = vld1q_dup_f32(A++);
		c[6] =  vfmaq_f32(c[6], b1, a);
		c[7] =  vfmaq_f32(c[7], b2, a);
		c[8] =  vfmaq_f32(c[8], b3, a);

		a = vld1q_dup_f32(A++);
		c[9] =  vfmaq_f32(c[9], b1, a);
		c[10] =  vfmaq_f32(c[10], b2, a);
		c[11] =  vfmaq_f32(c[11], b3, a);

		a = vld1q_dup_f32(A++);
		c[12] =  vfmaq_f32(c[12], b1, a);
		c[13] =  vfmaq_f32(c[13], b2, a);
		c[14] =  vfmaq_f32(c[14], b3, a);

		a = vld1q_dup_f32(A++);
		c[15] =  vfmaq_f32(c[15], b1, a);
		c[16] =  vfmaq_f32(c[16], b2, a);
		c[17] =  vfmaq_f32(c[17], b3, a);

		a = vld1q_dup_f32(A++);
		c[18] =  vfmaq_f32(c[18], b1, a);
		c[19] =  vfmaq_f32(c[19], b2, a);
		c[20] =  vfmaq_f32(c[20], b3, a);

		a = vld1q_dup_f32(A++);
		c[21] =  vfmaq_f32(c[21], b1, a);
		c[22] =  vfmaq_f32(c[22], b2, a);
		c[23] =  vfmaq_f32(c[23], b3, a);

		a = vld1q_dup_f32(A++);
		c[24] =  vfmaq_f32(c[24], b1, a);
		c[25] =  vfmaq_f32(c[25], b2, a);
		c[26] =  vfmaq_f32(c[26], b3, a);

		a = vld1q_dup_f32(A++);
		c[27] =  vfmaq_f32(c[27], b1, a);
		c[28] =  vfmaq_f32(c[28], b2, a);
		c[29] =  vfmaq_f32(c[29], b3, a);

		B += n;


		b1 = vld1q_f32(B);
		b2 = vld1q_f32(B + 4);
		b3 = vld1q_f32(B + 8);

		a = vld1q_dup_f32(A++);
		c[0] =  vfmaq_f32(c[0], b1, a);
		c[1] =  vfmaq_f32(c[1], b2, a);
		c[2] =  vfmaq_f32(c[2], b3, a);

		a = vld1q_dup_f32(A++);
		c[3] =  vfmaq_f32(c[3], b1, a);
		c[4] =  vfmaq_f32(c[4], b2, a);
		c[5] =  vfmaq_f32(c[5], b3, a);

		a = vld1q_dup_f32(A++);
		c[6] =  vfmaq_f32(c[6], b1, a);
		c[7] =  vfmaq_f32(c[7], b2, a);
		c[8] =  vfmaq_f32(c[8], b3, a);

		a = vld1q_dup_f32(A++);
		c[9] =  vfmaq_f32(c[9], b1, a);
		c[10] =  vfmaq_f32(c[10], b2, a);
		c[11] =  vfmaq_f32(c[11], b3, a);

		a = vld1q_dup_f32(A++);
		c[12] =  vfmaq_f32(c[12], b1, a);
		c[13] =  vfmaq_f32(c[13], b2, a);
		c[14] =  vfmaq_f32(c[14], b3, a);

		a = vld1q_dup_f32(A++);
		c[15] =  vfmaq_f32(c[15], b1, a);
		c[16] =  vfmaq_f32(c[16], b2, a);
		c[17] =  vfmaq_f32(c[17], b3, a);

		a = vld1q_dup_f32(A++);
		c[18] =  vfmaq_f32(c[18], b1, a);
		c[19] =  vfmaq_f32(c[19], b2, a);
		c[20] =  vfmaq_f32(c[20], b3, a);

		a = vld1q_dup_f32(A++);
		c[21] =  vfmaq_f32(c[21], b1, a);
		c[22] =  vfmaq_f32(c[22], b2, a);
		c[23] =  vfmaq_f32(c[23], b3, a);

		a = vld1q_dup_f32(A++);
		c[24] =  vfmaq_f32(c[24], b1, a);
		c[25] =  vfmaq_f32(c[25], b2, a);
		c[26] =  vfmaq_f32(c[26], b3, a);

		a = vld1q_dup_f32(A++);
		c[27] =  vfmaq_f32(c[27], b1, a);
		c[28] =  vfmaq_f32(c[28], b2, a);
		c[29] =  vfmaq_f32(c[29], b3, a);

		B += n;


		b1 = vld1q_f32(B);
		b2 = vld1q_f32(B + 4);
		b3 = vld1q_f32(B + 8);

		a = vld1q_dup_f32(A++);
		c[0] =  vfmaq_f32(c[0], b1, a);
		c[1] =  vfmaq_f32(c[1], b2, a);
		c[2] =  vfmaq_f32(c[2], b3, a);

		a = vld1q_dup_f32(A++);
		c[3] =  vfmaq_f32(c[3], b1, a);
		c[4] =  vfmaq_f32(c[4], b2, a);
		c[5] =  vfmaq_f32(c[5], b3, a);

		a = vld1q_dup_f32(A++);
		c[6] =  vfmaq_f32(c[6], b1, a);
		c[7] =  vfmaq_f32(c[7], b2, a);
		c[8] =  vfmaq_f32(c[8], b3, a);

		a = vld1q_dup_f32(A++);
		c[9] =  vfmaq_f32(c[9], b1, a);
		c[10] =  vfmaq_f32(c[10], b2, a);
		c[11] =  vfmaq_f32(c[11], b3, a);

		a = vld1q_dup_f32(A++);
		c[12] =  vfmaq_f32(c[12], b1, a);
		c[13] =  vfmaq_f32(c[13], b2, a);
		c[14] =  vfmaq_f32(c[14], b3, a);

		a = vld1q_dup_f32(A++);
		c[15] =  vfmaq_f32(c[15], b1, a);
		c[16] =  vfmaq_f32(c[16], b2, a);
		c[17] =  vfmaq_f32(c[17], b3, a);

		a = vld1q_dup_f32(A++);
		c[18] =  vfmaq_f32(c[18], b1, a);
		c[19] =  vfmaq_f32(c[19], b2, a);
		c[20] =  vfmaq_f32(c[20], b3, a);

		a = vld1q_dup_f32(A++);
		c[21] =  vfmaq_f32(c[21], b1, a);
		c[22] =  vfmaq_f32(c[22], b2, a);
		c[23] =  vfmaq_f32(c[23], b3, a);

		a = vld1q_dup_f32(A++);
		c[24] =  vfmaq_f32(c[24], b1, a);
		c[25] =  vfmaq_f32(c[25], b2, a);
		c[26] =  vfmaq_f32(c[26], b3, a);

		a = vld1q_dup_f32(A++);
		c[27] =  vfmaq_f32(c[27], b1, a);
		c[28] =  vfmaq_f32(c[28], b2, a);
		c[29] =  vfmaq_f32(c[29], b3, a);

		B += n;


		b1 = vld1q_f32(B);
		b2 = vld1q_f32(B + 4);
		b3 = vld1q_f32(B + 8);

		a = vld1q_dup_f32(A++);
		c[0] =  vfmaq_f32(c[0], b1, a);
		c[1] =  vfmaq_f32(c[1], b2, a);
		c[2] =  vfmaq_f32(c[2], b3, a);

		a = vld1q_dup_f32(A++);
		c[3] =  vfmaq_f32(c[3], b1, a);
		c[4] =  vfmaq_f32(c[4], b2, a);
		c[5] =  vfmaq_f32(c[5], b3, a);

		a = vld1q_dup_f32(A++);
		c[6] =  vfmaq_f32(c[6], b1, a);
		c[7] =  vfmaq_f32(c[7], b2, a);
		c[8] =  vfmaq_f32(c[8], b3, a);

		a = vld1q_dup_f32(A++);
		c[9] =  vfmaq_f32(c[9], b1, a);
		c[10] =  vfmaq_f32(c[10], b2, a);
		c[11] =  vfmaq_f32(c[11], b3, a);

		a = vld1q_dup_f32(A++);
		c[12] =  vfmaq_f32(c[12], b1, a);
		c[13] =  vfmaq_f32(c[13], b2, a);
		c[14] =  vfmaq_f32(c[14], b3, a);

		a = vld1q_dup_f32(A++);
		c[15] =  vfmaq_f32(c[15], b1, a);
		c[16] =  vfmaq_f32(c[16], b2, a);
		c[17] =  vfmaq_f32(c[17], b3, a);

		a = vld1q_dup_f32(A++);
		c[18] =  vfmaq_f32(c[18], b1, a);
		c[19] =  vfmaq_f32(c[19], b2, a);
		c[20] =  vfmaq_f32(c[20], b3, a);

		a = vld1q_dup_f32(A++);
		c[21] =  vfmaq_f32(c[21], b1, a);
		c[22] =  vfmaq_f32(c[22], b2, a);
		c[23] =  vfmaq_f32(c[23], b3, a);

		a = vld1q_dup_f32(A++);
		c[24] =  vfmaq_f32(c[24], b1, a);
		c[25] =  vfmaq_f32(c[25], b2, a);
		c[26] =  vfmaq_f32(c[26], b3, a);

		a = vld1q_dup_f32(A++);
		c[27] =  vfmaq_f32(c[27], b1, a);
		c[28] =  vfmaq_f32(c[28], b2, a);
		c[29] =  vfmaq_f32(c[29], b3, a);

		B += n;


	}
			
	for(int kk = 0; kk < rem; kk++) { 
			
		b1 = vld1q_f32(B);
		b2 = vld1q_f32(B + 4);
		b3 = vld1q_f32(B + 8);

		a = vld1q_dup_f32(A++);
		c[0] =  vfmaq_f32(c[0], b1, a);
		c[1] =  vfmaq_f32(c[1], b2, a);
		c[2] =  vfmaq_f32(c[2], b3, a);

		a = vld1q_dup_f32(A++);
		c[3] =  vfmaq_f32(c[3], b1, a);
		c[4] =  vfmaq_f32(c[4], b2, a);
		c[5] =  vfmaq_f32(c[5], b3, a);

		a = vld1q_dup_f32(A++);
		c[6] =  vfmaq_f32(c[6], b1, a);
		c[7] =  vfmaq_f32(c[7], b2, a);
		c[8] =  vfmaq_f32(c[8], b3, a);

		a = vld1q_dup_f32(A++);
		c[9] =  vfmaq_f32(c[9], b1, a);
		c[10] =  vfmaq_f32(c[10], b2, a);
		c[11] =  vfmaq_f32(c[11], b3, a);

		a = vld1q_dup_f32(A++);
		c[12] =  vfmaq_f32(c[12], b1, a);
		c[13] =  vfmaq_f32(c[13], b2, a);
		c[14] =  vfmaq_f32(c[14], b3, a);

		a = vld1q_dup_f32(A++);
		c[15] =  vfmaq_f32(c[15], b1, a);
		c[16] =  vfmaq_f32(c[16], b2, a);
		c[17] =  vfmaq_f32(c[17], b3, a);

		a = vld1q_dup_f32(A++);
		c[18] =  vfmaq_f32(c[18], b1, a);
		c[19] =  vfmaq_f32(c[19], b2, a);
		c[20] =  vfmaq_f32(c[20], b3, a);

		a = vld1q_dup_f32(A++);
		c[21] =  vfmaq_f32(c[21], b1, a);
		c[22] =  vfmaq_f32(c[22], b2, a);
		c[23] =  vfmaq_f32(c[23], b3, a);

		a = vld1q_dup_f32(A++);
		c[24] =  vfmaq_f32(c[24], b1, a);
		c[25] =  vfmaq_f32(c[25], b2, a);
		c[26] =  vfmaq_f32(c[26], b3, a);

		a = vld1q_dup_f32(A++);
		c[27] =  vfmaq_f32(c[27], b1, a);
		c[28] =  vfmaq_f32(c[28], b2, a);
		c[29] =  vfmaq_f32(c[29], b3, a);

		B += n;
			}
			
	vst1q_f32(C, c[0]);
	vst1q_f32(C + 4, c[1]);
	vst1q_f32(C + 8, c[2]);
	vst1q_f32(C + 12, c[3]);
	vst1q_f32(C + 16, c[4]);
	vst1q_f32(C + 20, c[5]);
	vst1q_f32(C + 24, c[6]);
	vst1q_f32(C + 28, c[7]);
	vst1q_f32(C + 32, c[8]);
	vst1q_f32(C + 36, c[9]);
	vst1q_f32(C + 40, c[10]);
	vst1q_f32(C + 44, c[11]);
	vst1q_f32(C + 48, c[12]);
	vst1q_f32(C + 52, c[13]);
	vst1q_f32(C + 56, c[14]);
	vst1q_f32(C + 60, c[15]);
	vst1q_f32(C + 64, c[16]);
	vst1q_f32(C + 68, c[17]);
	vst1q_f32(C + 72, c[18]);
	vst1q_f32(C + 76, c[19]);
	vst1q_f32(C + 80, c[20]);
	vst1q_f32(C + 84, c[21]);
	vst1q_f32(C + 88, c[22]);
	vst1q_f32(C + 92, c[23]);
	vst1q_f32(C + 96, c[24]);
	vst1q_f32(C + 100, c[25]);
	vst1q_f32(C + 104, c[26]);
	vst1q_f32(C + 108, c[27]);
	vst1q_f32(C + 112, c[28]);
	vst1q_f32(C + 116, c[29]);
}
void cake_sgemm_armv8_10x24(float* A, float* B, float* C, int m, int n, int k) {
			  
	float32x4_t a, b1, b2, b3, b4, b5, b6;
	float32x4_t c[10*6];
	
	// load tile of C into arm neon SIMD registers
	c[0]  = vld1q_f32(C);
	c[1]  = vld1q_f32(C + 4);
	c[2]  = vld1q_f32(C + 8);
	c[3]  = vld1q_f32(C + 12);
	c[4]  = vld1q_f32(C + 16);
	c[5]  = vld1q_f32(C + 20);
	c[6]  = vld1q_f32(C + 24);
	c[7]  = vld1q_f32(C + 28);
	c[8]  = vld1q_f32(C + 32);
	c[9]  = vld1q_f32(C + 36);
	c[10]  = vld1q_f32(C + 40);
	c[11]  = vld1q_f32(C + 44);
	c[12]  = vld1q_f32(C + 48);
	c[13]  = vld1q_f32(C + 52);
	c[14]  = vld1q_f32(C + 56);
	c[15]  = vld1q_f32(C + 60);
	c[16]  = vld1q_f32(C + 64);
	c[17]  = vld1q_f32(C + 68);
	c[18]  = vld1q_f32(C + 72);
	c[19]  = vld1q_f32(C + 76);
	c[20]  = vld1q_f32(C + 80);
	c[21]  = vld1q_f32(C + 84);
	c[22]  = vld1q_f32(C + 88);
	c[23]  = vld1q_f32(C + 92);
	c[24]  = vld1q_f32(C + 96);
	c[25]  = vld1q_f32(C + 100);
	c[26]  = vld1q_f32(C + 104);
	c[27]  = vld1q_f32(C + 108);
	c[28]  = vld1q_f32(C + 112);
	c[29]  = vld1q_f32(C + 116);
	c[30]  = vld1q_f32(C + 120);
	c[31]  = vld1q_f32(C + 124);
	c[32]  = vld1q_f32(C + 128);
	c[33]  = vld1q_f32(C + 132);
	c[34]  = vld1q_f32(C + 136);
	c[35]  = vld1q_f32(C + 140);
	c[36]  = vld1q_f32(C + 144);
	c[37]  = vld1q_f32(C + 148);
	c[38]  = vld1q_f32(C + 152);
	c[39]  = vld1q_f32(C + 156);
	c[40]  = vld1q_f32(C + 160);
	c[41]  = vld1q_f32(C + 164);
	c[42]  = vld1q_f32(C + 168);
	c[43]  = vld1q_f32(C + 172);
	c[44]  = vld1q_f32(C + 176);
	c[45]  = vld1q_f32(C + 180);
	c[46]  = vld1q_f32(C + 184);
	c[47]  = vld1q_f32(C + 188);
	c[48]  = vld1q_f32(C + 192);
	c[49]  = vld1q_f32(C + 196);
	c[50]  = vld1q_f32(C + 200);
	c[51]  = vld1q_f32(C + 204);
	c[52]  = vld1q_f32(C + 208);
	c[53]  = vld1q_f32(C + 212);
	c[54]  = vld1q_f32(C + 216);
	c[55]  = vld1q_f32(C + 220);
	c[56]  = vld1q_f32(C + 224);
	c[57]  = vld1q_f32(C + 228);
	c[58]  = vld1q_f32(C + 232);
	c[59]  = vld1q_f32(C + 236);

	int rem = k % 4;
	k -= rem;

	// outer-product unrolled 4 times
	for(int kk = 0; kk < k; kk += 4) { 

			
		b1 = vld1q_f32(B);
		b2 = vld1q_f32(B + 4);
		b3 = vld1q_f32(B + 8);
		b4 = vld1q_f32(B + 12);
		b5 = vld1q_f32(B + 16);
		b6 = vld1q_f32(B + 20);

		a = vld1q_dup_f32(A++);
		c[0] =  vfmaq_f32(c[0], b1, a);
		c[1] =  vfmaq_f32(c[1], b2, a);
		c[2] =  vfmaq_f32(c[2], b3, a);
		c[3] =  vfmaq_f32(c[3], b4, a);
		c[4] =  vfmaq_f32(c[4], b5, a);
		c[5] =  vfmaq_f32(c[5], b6, a);

		a = vld1q_dup_f32(A++);
		c[6] =  vfmaq_f32(c[6], b1, a);
		c[7] =  vfmaq_f32(c[7], b2, a);
		c[8] =  vfmaq_f32(c[8], b3, a);
		c[9] =  vfmaq_f32(c[9], b4, a);
		c[10] =  vfmaq_f32(c[10], b5, a);
		c[11] =  vfmaq_f32(c[11], b6, a);

		a = vld1q_dup_f32(A++);
		c[12] =  vfmaq_f32(c[12], b1, a);
		c[13] =  vfmaq_f32(c[13], b2, a);
		c[14] =  vfmaq_f32(c[14], b3, a);
		c[15] =  vfmaq_f32(c[15], b4, a);
		c[16] =  vfmaq_f32(c[16], b5, a);
		c[17] =  vfmaq_f32(c[17], b6, a);

		a = vld1q_dup_f32(A++);
		c[18] =  vfmaq_f32(c[18], b1, a);
		c[19] =  vfmaq_f32(c[19], b2, a);
		c[20] =  vfmaq_f32(c[20], b3, a);
		c[21] =  vfmaq_f32(c[21], b4, a);
		c[22] =  vfmaq_f32(c[22], b5, a);
		c[23] =  vfmaq_f32(c[23], b6, a);

		a = vld1q_dup_f32(A++);
		c[24] =  vfmaq_f32(c[24], b1, a);
		c[25] =  vfmaq_f32(c[25], b2, a);
		c[26] =  vfmaq_f32(c[26], b3, a);
		c[27] =  vfmaq_f32(c[27], b4, a);
		c[28] =  vfmaq_f32(c[28], b5, a);
		c[29] =  vfmaq_f32(c[29], b6, a);

		a = vld1q_dup_f32(A++);
		c[30] =  vfmaq_f32(c[30], b1, a);
		c[31] =  vfmaq_f32(c[31], b2, a);
		c[32] =  vfmaq_f32(c[32], b3, a);
		c[33] =  vfmaq_f32(c[33], b4, a);
		c[34] =  vfmaq_f32(c[34], b5, a);
		c[35] =  vfmaq_f32(c[35], b6, a);

		a = vld1q_dup_f32(A++);
		c[36] =  vfmaq_f32(c[36], b1, a);
		c[37] =  vfmaq_f32(c[37], b2, a);
		c[38] =  vfmaq_f32(c[38], b3, a);
		c[39] =  vfmaq_f32(c[39], b4, a);
		c[40] =  vfmaq_f32(c[40], b5, a);
		c[41] =  vfmaq_f32(c[41], b6, a);

		a = vld1q_dup_f32(A++);
		c[42] =  vfmaq_f32(c[42], b1, a);
		c[43] =  vfmaq_f32(c[43], b2, a);
		c[44] =  vfmaq_f32(c[44], b3, a);
		c[45] =  vfmaq_f32(c[45], b4, a);
		c[46] =  vfmaq_f32(c[46], b5, a);
		c[47] =  vfmaq_f32(c[47], b6, a);

		a = vld1q_dup_f32(A++);
		c[48] =  vfmaq_f32(c[48], b1, a);
		c[49] =  vfmaq_f32(c[49], b2, a);
		c[50] =  vfmaq_f32(c[50], b3, a);
		c[51] =  vfmaq_f32(c[51], b4, a);
		c[52] =  vfmaq_f32(c[52], b5, a);
		c[53] =  vfmaq_f32(c[53], b6, a);

		a = vld1q_dup_f32(A++);
		c[54] =  vfmaq_f32(c[54], b1, a);
		c[55] =  vfmaq_f32(c[55], b2, a);
		c[56] =  vfmaq_f32(c[56], b3, a);
		c[57] =  vfmaq_f32(c[57], b4, a);
		c[58] =  vfmaq_f32(c[58], b5, a);
		c[59] =  vfmaq_f32(c[59], b6, a);

		B += n;


		b1 = vld1q_f32(B);
		b2 = vld1q_f32(B + 4);
		b3 = vld1q_f32(B + 8);
		b4 = vld1q_f32(B + 12);
		b5 = vld1q_f32(B + 16);
		b6 = vld1q_f32(B + 20);

		a = vld1q_dup_f32(A++);
		c[0] =  vfmaq_f32(c[0], b1, a);
		c[1] =  vfmaq_f32(c[1], b2, a);
		c[2] =  vfmaq_f32(c[2], b3, a);
		c[3] =  vfmaq_f32(c[3], b4, a);
		c[4] =  vfmaq_f32(c[4], b5, a);
		c[5] =  vfmaq_f32(c[5], b6, a);

		a = vld1q_dup_f32(A++);
		c[6] =  vfmaq_f32(c[6], b1, a);
		c[7] =  vfmaq_f32(c[7], b2, a);
		c[8] =  vfmaq_f32(c[8], b3, a);
		c[9] =  vfmaq_f32(c[9], b4, a);
		c[10] =  vfmaq_f32(c[10], b5, a);
		c[11] =  vfmaq_f32(c[11], b6, a);

		a = vld1q_dup_f32(A++);
		c[12] =  vfmaq_f32(c[12], b1, a);
		c[13] =  vfmaq_f32(c[13], b2, a);
		c[14] =  vfmaq_f32(c[14], b3, a);
		c[15] =  vfmaq_f32(c[15], b4, a);
		c[16] =  vfmaq_f32(c[16], b5, a);
		c[17] =  vfmaq_f32(c[17], b6, a);

		a = vld1q_dup_f32(A++);
		c[18] =  vfmaq_f32(c[18], b1, a);
		c[19] =  vfmaq_f32(c[19], b2, a);
		c[20] =  vfmaq_f32(c[20], b3, a);
		c[21] =  vfmaq_f32(c[21], b4, a);
		c[22] =  vfmaq_f32(c[22], b5, a);
		c[23] =  vfmaq_f32(c[23], b6, a);

		a = vld1q_dup_f32(A++);
		c[24] =  vfmaq_f32(c[24], b1, a);
		c[25] =  vfmaq_f32(c[25], b2, a);
		c[26] =  vfmaq_f32(c[26], b3, a);
		c[27] =  vfmaq_f32(c[27], b4, a);
		c[28] =  vfmaq_f32(c[28], b5, a);
		c[29] =  vfmaq_f32(c[29], b6, a);

		a = vld1q_dup_f32(A++);
		c[30] =  vfmaq_f32(c[30], b1, a);
		c[31] =  vfmaq_f32(c[31], b2, a);
		c[32] =  vfmaq_f32(c[32], b3, a);
		c[33] =  vfmaq_f32(c[33], b4, a);
		c[34] =  vfmaq_f32(c[34], b5, a);
		c[35] =  vfmaq_f32(c[35], b6, a);

		a = vld1q_dup_f32(A++);
		c[36] =  vfmaq_f32(c[36], b1, a);
		c[37] =  vfmaq_f32(c[37], b2, a);
		c[38] =  vfmaq_f32(c[38], b3, a);
		c[39] =  vfmaq_f32(c[39], b4, a);
		c[40] =  vfmaq_f32(c[40], b5, a);
		c[41] =  vfmaq_f32(c[41], b6, a);

		a = vld1q_dup_f32(A++);
		c[42] =  vfmaq_f32(c[42], b1, a);
		c[43] =  vfmaq_f32(c[43], b2, a);
		c[44] =  vfmaq_f32(c[44], b3, a);
		c[45] =  vfmaq_f32(c[45], b4, a);
		c[46] =  vfmaq_f32(c[46], b5, a);
		c[47] =  vfmaq_f32(c[47], b6, a);

		a = vld1q_dup_f32(A++);
		c[48] =  vfmaq_f32(c[48], b1, a);
		c[49] =  vfmaq_f32(c[49], b2, a);
		c[50] =  vfmaq_f32(c[50], b3, a);
		c[51] =  vfmaq_f32(c[51], b4, a);
		c[52] =  vfmaq_f32(c[52], b5, a);
		c[53] =  vfmaq_f32(c[53], b6, a);

		a = vld1q_dup_f32(A++);
		c[54] =  vfmaq_f32(c[54], b1, a);
		c[55] =  vfmaq_f32(c[55], b2, a);
		c[56] =  vfmaq_f32(c[56], b3, a);
		c[57] =  vfmaq_f32(c[57], b4, a);
		c[58] =  vfmaq_f32(c[58], b5, a);
		c[59] =  vfmaq_f32(c[59], b6, a);

		B += n;


		b1 = vld1q_f32(B);
		b2 = vld1q_f32(B + 4);
		b3 = vld1q_f32(B + 8);
		b4 = vld1q_f32(B + 12);
		b5 = vld1q_f32(B + 16);
		b6 = vld1q_f32(B + 20);

		a = vld1q_dup_f32(A++);
		c[0] =  vfmaq_f32(c[0], b1, a);
		c[1] =  vfmaq_f32(c[1], b2, a);
		c[2] =  vfmaq_f32(c[2], b3, a);
		c[3] =  vfmaq_f32(c[3], b4, a);
		c[4] =  vfmaq_f32(c[4], b5, a);
		c[5] =  vfmaq_f32(c[5], b6, a);

		a = vld1q_dup_f32(A++);
		c[6] =  vfmaq_f32(c[6], b1, a);
		c[7] =  vfmaq_f32(c[7], b2, a);
		c[8] =  vfmaq_f32(c[8], b3, a);
		c[9] =  vfmaq_f32(c[9], b4, a);
		c[10] =  vfmaq_f32(c[10], b5, a);
		c[11] =  vfmaq_f32(c[11], b6, a);

		a = vld1q_dup_f32(A++);
		c[12] =  vfmaq_f32(c[12], b1, a);
		c[13] =  vfmaq_f32(c[13], b2, a);
		c[14] =  vfmaq_f32(c[14], b3, a);
		c[15] =  vfmaq_f32(c[15], b4, a);
		c[16] =  vfmaq_f32(c[16], b5, a);
		c[17] =  vfmaq_f32(c[17], b6, a);

		a = vld1q_dup_f32(A++);
		c[18] =  vfmaq_f32(c[18], b1, a);
		c[19] =  vfmaq_f32(c[19], b2, a);
		c[20] =  vfmaq_f32(c[20], b3, a);
		c[21] =  vfmaq_f32(c[21], b4, a);
		c[22] =  vfmaq_f32(c[22], b5, a);
		c[23] =  vfmaq_f32(c[23], b6, a);

		a = vld1q_dup_f32(A++);
		c[24] =  vfmaq_f32(c[24], b1, a);
		c[25] =  vfmaq_f32(c[25], b2, a);
		c[26] =  vfmaq_f32(c[26], b3, a);
		c[27] =  vfmaq_f32(c[27], b4, a);
		c[28] =  vfmaq_f32(c[28], b5, a);
		c[29] =  vfmaq_f32(c[29], b6, a);

		a = vld1q_dup_f32(A++);
		c[30] =  vfmaq_f32(c[30], b1, a);
		c[31] =  vfmaq_f32(c[31], b2, a);
		c[32] =  vfmaq_f32(c[32], b3, a);
		c[33] =  vfmaq_f32(c[33], b4, a);
		c[34] =  vfmaq_f32(c[34], b5, a);
		c[35] =  vfmaq_f32(c[35], b6, a);

		a = vld1q_dup_f32(A++);
		c[36] =  vfmaq_f32(c[36], b1, a);
		c[37] =  vfmaq_f32(c[37], b2, a);
		c[38] =  vfmaq_f32(c[38], b3, a);
		c[39] =  vfmaq_f32(c[39], b4, a);
		c[40] =  vfmaq_f32(c[40], b5, a);
		c[41] =  vfmaq_f32(c[41], b6, a);

		a = vld1q_dup_f32(A++);
		c[42] =  vfmaq_f32(c[42], b1, a);
		c[43] =  vfmaq_f32(c[43], b2, a);
		c[44] =  vfmaq_f32(c[44], b3, a);
		c[45] =  vfmaq_f32(c[45], b4, a);
		c[46] =  vfmaq_f32(c[46], b5, a);
		c[47] =  vfmaq_f32(c[47], b6, a);

		a = vld1q_dup_f32(A++);
		c[48] =  vfmaq_f32(c[48], b1, a);
		c[49] =  vfmaq_f32(c[49], b2, a);
		c[50] =  vfmaq_f32(c[50], b3, a);
		c[51] =  vfmaq_f32(c[51], b4, a);
		c[52] =  vfmaq_f32(c[52], b5, a);
		c[53] =  vfmaq_f32(c[53], b6, a);

		a = vld1q_dup_f32(A++);
		c[54] =  vfmaq_f32(c[54], b1, a);
		c[55] =  vfmaq_f32(c[55], b2, a);
		c[56] =  vfmaq_f32(c[56], b3, a);
		c[57] =  vfmaq_f32(c[57], b4, a);
		c[58] =  vfmaq_f32(c[58], b5, a);
		c[59] =  vfmaq_f32(c[59], b6, a);

		B += n;


		b1 = vld1q_f32(B);
		b2 = vld1q_f32(B + 4);
		b3 = vld1q_f32(B + 8);
		b4 = vld1q_f32(B + 12);
		b5 = vld1q_f32(B + 16);
		b6 = vld1q_f32(B + 20);

		a = vld1q_dup_f32(A++);
		c[0] =  vfmaq_f32(c[0], b1, a);
		c[1] =  vfmaq_f32(c[1], b2, a);
		c[2] =  vfmaq_f32(c[2], b3, a);
		c[3] =  vfmaq_f32(c[3], b4, a);
		c[4] =  vfmaq_f32(c[4], b5, a);
		c[5] =  vfmaq_f32(c[5], b6, a);

		a = vld1q_dup_f32(A++);
		c[6] =  vfmaq_f32(c[6], b1, a);
		c[7] =  vfmaq_f32(c[7], b2, a);
		c[8] =  vfmaq_f32(c[8], b3, a);
		c[9] =  vfmaq_f32(c[9], b4, a);
		c[10] =  vfmaq_f32(c[10], b5, a);
		c[11] =  vfmaq_f32(c[11], b6, a);

		a = vld1q_dup_f32(A++);
		c[12] =  vfmaq_f32(c[12], b1, a);
		c[13] =  vfmaq_f32(c[13], b2, a);
		c[14] =  vfmaq_f32(c[14], b3, a);
		c[15] =  vfmaq_f32(c[15], b4, a);
		c[16] =  vfmaq_f32(c[16], b5, a);
		c[17] =  vfmaq_f32(c[17], b6, a);

		a = vld1q_dup_f32(A++);
		c[18] =  vfmaq_f32(c[18], b1, a);
		c[19] =  vfmaq_f32(c[19], b2, a);
		c[20] =  vfmaq_f32(c[20], b3, a);
		c[21] =  vfmaq_f32(c[21], b4, a);
		c[22] =  vfmaq_f32(c[22], b5, a);
		c[23] =  vfmaq_f32(c[23], b6, a);

		a = vld1q_dup_f32(A++);
		c[24] =  vfmaq_f32(c[24], b1, a);
		c[25] =  vfmaq_f32(c[25], b2, a);
		c[26] =  vfmaq_f32(c[26], b3, a);
		c[27] =  vfmaq_f32(c[27], b4, a);
		c[28] =  vfmaq_f32(c[28], b5, a);
		c[29] =  vfmaq_f32(c[29], b6, a);

		a = vld1q_dup_f32(A++);
		c[30] =  vfmaq_f32(c[30], b1, a);
		c[31] =  vfmaq_f32(c[31], b2, a);
		c[32] =  vfmaq_f32(c[32], b3, a);
		c[33] =  vfmaq_f32(c[33], b4, a);
		c[34] =  vfmaq_f32(c[34], b5, a);
		c[35] =  vfmaq_f32(c[35], b6, a);

		a = vld1q_dup_f32(A++);
		c[36] =  vfmaq_f32(c[36], b1, a);
		c[37] =  vfmaq_f32(c[37], b2, a);
		c[38] =  vfmaq_f32(c[38], b3, a);
		c[39] =  vfmaq_f32(c[39], b4, a);
		c[40] =  vfmaq_f32(c[40], b5, a);
		c[41] =  vfmaq_f32(c[41], b6, a);

		a = vld1q_dup_f32(A++);
		c[42] =  vfmaq_f32(c[42], b1, a);
		c[43] =  vfmaq_f32(c[43], b2, a);
		c[44] =  vfmaq_f32(c[44], b3, a);
		c[45] =  vfmaq_f32(c[45], b4, a);
		c[46] =  vfmaq_f32(c[46], b5, a);
		c[47] =  vfmaq_f32(c[47], b6, a);

		a = vld1q_dup_f32(A++);
		c[48] =  vfmaq_f32(c[48], b1, a);
		c[49] =  vfmaq_f32(c[49], b2, a);
		c[50] =  vfmaq_f32(c[50], b3, a);
		c[51] =  vfmaq_f32(c[51], b4, a);
		c[52] =  vfmaq_f32(c[52], b5, a);
		c[53] =  vfmaq_f32(c[53], b6, a);

		a = vld1q_dup_f32(A++);
		c[54] =  vfmaq_f32(c[54], b1, a);
		c[55] =  vfmaq_f32(c[55], b2, a);
		c[56] =  vfmaq_f32(c[56], b3, a);
		c[57] =  vfmaq_f32(c[57], b4, a);
		c[58] =  vfmaq_f32(c[58], b5, a);
		c[59] =  vfmaq_f32(c[59], b6, a);

		B += n;


	}
			
	for(int kk = 0; kk < rem; kk++) { 
			
		b1 = vld1q_f32(B);
		b2 = vld1q_f32(B + 4);
		b3 = vld1q_f32(B + 8);
		b4 = vld1q_f32(B + 12);
		b5 = vld1q_f32(B + 16);
		b6 = vld1q_f32(B + 20);

		a = vld1q_dup_f32(A++);
		c[0] =  vfmaq_f32(c[0], b1, a);
		c[1] =  vfmaq_f32(c[1], b2, a);
		c[2] =  vfmaq_f32(c[2], b3, a);
		c[3] =  vfmaq_f32(c[3], b4, a);
		c[4] =  vfmaq_f32(c[4], b5, a);
		c[5] =  vfmaq_f32(c[5], b6, a);

		a = vld1q_dup_f32(A++);
		c[6] =  vfmaq_f32(c[6], b1, a);
		c[7] =  vfmaq_f32(c[7], b2, a);
		c[8] =  vfmaq_f32(c[8], b3, a);
		c[9] =  vfmaq_f32(c[9], b4, a);
		c[10] =  vfmaq_f32(c[10], b5, a);
		c[11] =  vfmaq_f32(c[11], b6, a);

		a = vld1q_dup_f32(A++);
		c[12] =  vfmaq_f32(c[12], b1, a);
		c[13] =  vfmaq_f32(c[13], b2, a);
		c[14] =  vfmaq_f32(c[14], b3, a);
		c[15] =  vfmaq_f32(c[15], b4, a);
		c[16] =  vfmaq_f32(c[16], b5, a);
		c[17] =  vfmaq_f32(c[17], b6, a);

		a = vld1q_dup_f32(A++);
		c[18] =  vfmaq_f32(c[18], b1, a);
		c[19] =  vfmaq_f32(c[19], b2, a);
		c[20] =  vfmaq_f32(c[20], b3, a);
		c[21] =  vfmaq_f32(c[21], b4, a);
		c[22] =  vfmaq_f32(c[22], b5, a);
		c[23] =  vfmaq_f32(c[23], b6, a);

		a = vld1q_dup_f32(A++);
		c[24] =  vfmaq_f32(c[24], b1, a);
		c[25] =  vfmaq_f32(c[25], b2, a);
		c[26] =  vfmaq_f32(c[26], b3, a);
		c[27] =  vfmaq_f32(c[27], b4, a);
		c[28] =  vfmaq_f32(c[28], b5, a);
		c[29] =  vfmaq_f32(c[29], b6, a);

		a = vld1q_dup_f32(A++);
		c[30] =  vfmaq_f32(c[30], b1, a);
		c[31] =  vfmaq_f32(c[31], b2, a);
		c[32] =  vfmaq_f32(c[32], b3, a);
		c[33] =  vfmaq_f32(c[33], b4, a);
		c[34] =  vfmaq_f32(c[34], b5, a);
		c[35] =  vfmaq_f32(c[35], b6, a);

		a = vld1q_dup_f32(A++);
		c[36] =  vfmaq_f32(c[36], b1, a);
		c[37] =  vfmaq_f32(c[37], b2, a);
		c[38] =  vfmaq_f32(c[38], b3, a);
		c[39] =  vfmaq_f32(c[39], b4, a);
		c[40] =  vfmaq_f32(c[40], b5, a);
		c[41] =  vfmaq_f32(c[41], b6, a);

		a = vld1q_dup_f32(A++);
		c[42] =  vfmaq_f32(c[42], b1, a);
		c[43] =  vfmaq_f32(c[43], b2, a);
		c[44] =  vfmaq_f32(c[44], b3, a);
		c[45] =  vfmaq_f32(c[45], b4, a);
		c[46] =  vfmaq_f32(c[46], b5, a);
		c[47] =  vfmaq_f32(c[47], b6, a);

		a = vld1q_dup_f32(A++);
		c[48] =  vfmaq_f32(c[48], b1, a);
		c[49] =  vfmaq_f32(c[49], b2, a);
		c[50] =  vfmaq_f32(c[50], b3, a);
		c[51] =  vfmaq_f32(c[51], b4, a);
		c[52] =  vfmaq_f32(c[52], b5, a);
		c[53] =  vfmaq_f32(c[53], b6, a);

		a = vld1q_dup_f32(A++);
		c[54] =  vfmaq_f32(c[54], b1, a);
		c[55] =  vfmaq_f32(c[55], b2, a);
		c[56] =  vfmaq_f32(c[56], b3, a);
		c[57] =  vfmaq_f32(c[57], b4, a);
		c[58] =  vfmaq_f32(c[58], b5, a);
		c[59] =  vfmaq_f32(c[59], b6, a);

		B += n;
			}
			
	vst1q_f32(C, c[0]);
	vst1q_f32(C + 4, c[1]);
	vst1q_f32(C + 8, c[2]);
	vst1q_f32(C + 12, c[3]);
	vst1q_f32(C + 16, c[4]);
	vst1q_f32(C + 20, c[5]);
	vst1q_f32(C + 24, c[6]);
	vst1q_f32(C + 28, c[7]);
	vst1q_f32(C + 32, c[8]);
	vst1q_f32(C + 36, c[9]);
	vst1q_f32(C + 40, c[10]);
	vst1q_f32(C + 44, c[11]);
	vst1q_f32(C + 48, c[12]);
	vst1q_f32(C + 52, c[13]);
	vst1q_f32(C + 56, c[14]);
	vst1q_f32(C + 60, c[15]);
	vst1q_f32(C + 64, c[16]);
	vst1q_f32(C + 68, c[17]);
	vst1q_f32(C + 72, c[18]);
	vst1q_f32(C + 76, c[19]);
	vst1q_f32(C + 80, c[20]);
	vst1q_f32(C + 84, c[21]);
	vst1q_f32(C + 88, c[22]);
	vst1q_f32(C + 92, c[23]);
	vst1q_f32(C + 96, c[24]);
	vst1q_f32(C + 100, c[25]);
	vst1q_f32(C + 104, c[26]);
	vst1q_f32(C + 108, c[27]);
	vst1q_f32(C + 112, c[28]);
	vst1q_f32(C + 116, c[29]);
	vst1q_f32(C + 120, c[30]);
	vst1q_f32(C + 124, c[31]);
	vst1q_f32(C + 128, c[32]);
	vst1q_f32(C + 132, c[33]);
	vst1q_f32(C + 136, c[34]);
	vst1q_f32(C + 140, c[35]);
	vst1q_f32(C + 144, c[36]);
	vst1q_f32(C + 148, c[37]);
	vst1q_f32(C + 152, c[38]);
	vst1q_f32(C + 156, c[39]);
	vst1q_f32(C + 160, c[40]);
	vst1q_f32(C + 164, c[41]);
	vst1q_f32(C + 168, c[42]);
	vst1q_f32(C + 172, c[43]);
	vst1q_f32(C + 176, c[44]);
	vst1q_f32(C + 180, c[45]);
	vst1q_f32(C + 184, c[46]);
	vst1q_f32(C + 188, c[47]);
	vst1q_f32(C + 192, c[48]);
	vst1q_f32(C + 196, c[49]);
	vst1q_f32(C + 200, c[50]);
	vst1q_f32(C + 204, c[51]);
	vst1q_f32(C + 208, c[52]);
	vst1q_f32(C + 212, c[53]);
	vst1q_f32(C + 216, c[54]);
	vst1q_f32(C + 220, c[55]);
	vst1q_f32(C + 224, c[56]);
	vst1q_f32(C + 228, c[57]);
	vst1q_f32(C + 232, c[58]);
	vst1q_f32(C + 236, c[59]);
}