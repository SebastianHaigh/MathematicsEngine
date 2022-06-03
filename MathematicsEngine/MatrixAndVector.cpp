#include "MathematicsEngine.h"

void add(Matrix33& out, const Matrix33& A, const Matrix33& B) {
	__m256 vec_a = _mm256_load_ps(&A.m[0]);
	__m256 vec_b = _mm256_load_ps(&B.m[0]);

	_mm256_store_ps(&out.m[0], _mm256_add_ps(vec_a, vec_b));
	out.m[8] += A.m[8] + B.m[8];

}

// Multiplies two 3x3 matrices and stores the result in the object that calls the method.
void multiply(Matrix33& out, const Matrix33& A, const Matrix33& B) {
	// a0 a1 a2     b0 b1 b2     c0 c1 c2
	// a3 a4 a5  *  b3 b4 b5  =  c3 c4 c5
	// a6 a7 a8     b6 b7 b8     c6 c7 c8
	//
	// c0 = a0*b0 + a1*b3 + a2*b6
	// c1 = a0*b1 + a1*b4 + a2*b7
	// c2 = a0*b2 + a1*b5 + a2*b8
	// 
	// c3 = a3*b0 + a4*b3 + a5*b6
	// c4 = a3*b1 + a4*b4 + a5*b7
	// c5 = a3*b2 + a4*b5 + a5*b8
	//
	// c6 = a6*b0 + a7*b3 + a8*b6
	// c7 = a6*b1 + a7*b4 + a8*b7
	// c8 = a6*b2 + a7*b5 + a8*b8

	// in the below we get:
	// row_0 = [ b0 b1 b2 b3(junk) ]
	// row_1 = [ b3 b4 b5 b6(junk) ]
	// row_2 = [ b6 b7 b8 b9(junk) ]
	__m128 row_0 = _mm_load_ps(&B.m[0]);
	__m128 row_1 = _mm_load_ps(&B.m[3]);
	__m128 row_2 = _mm_load_ps(&B.m[6]);

	// a_00 = [ a0 a0 a0 a0(junk) ]
	// a_01 = [ a1 a1 a1 a1(junk) ]
	// a_02 = [ a2 a2 a2 a2(junk) ]
	__m128 a_00 = _mm_broadcast_ss(&A.m[0]);
	__m128 a_01 = _mm_broadcast_ss(&A.m[1]);
	__m128 a_02 = _mm_broadcast_ss(&A.m[2]);

	// c_row_0 = [ a0*b0 a0*b1 a0*b2 junk ]
	__m128 c_row_0 = _mm_mul_ps(a_00, row_0);

	// temp = a_01 * row_1 = [ a1*b3 a1*b4 a1*b5 junk ]
	// c_row_0 + temp = [ a0*b0 + a1*b3, a0*b1 + a1*b4, a0*b2 + a1*b5, junk ]
	c_row_0 = _mm_fmadd_ps(a_01, row_1, c_row_0);

	// temp = a_02 * row_2 = [ a2*b6 a2*b7 a2*b8 junk ]
	// c_row_0 + temp = [ a0*b0 + a1*b3 + a2*b6, a0*b1 + a1*b4 + a2*b7, a0*b2 + a1*b5 + a2*b8, junk ]
	c_row_0 = _mm_fmadd_ps(a_02, row_2, c_row_0);

	// Put the result into the array c:
	// c = [ a0*b0 + a1*b3 + a2*b6, a0*b1 + a1*b4 + a2*b7, a0*b2 + a1*b5 + a2*b8, junk, 0.0 ..., 0.0]
	_mm_store_ps(&out.m[0], c_row_0);

	__m128 a_10 = _mm_broadcast_ss(&A.m[3]);
	__m128 a_11 = _mm_broadcast_ss(&A.m[4]);
	__m128 a_12 = _mm_broadcast_ss(&A.m[5]);

	__m128 c_row_1 = _mm_mul_ps(a_10, row_0);
	c_row_1 = _mm_fmadd_ps(a_11, row_1, c_row_1);
	c_row_1 = _mm_fmadd_ps(a_12, row_2, c_row_1);
	_mm_store_ps(&out.m[3], c_row_1); // This store operation will overwrite the junk that is currently in m[3], and place new junk in m[6]

	__m128 a_20 = _mm_broadcast_ss(&A.m[6]);
	__m128 a_21 = _mm_broadcast_ss(&A.m[7]);
	__m128 a_22 = _mm_broadcast_ss(&A.m[8]);

	__m128 c_row_2 = _mm_mul_ps(a_20, row_0);
	c_row_2 = _mm_fmadd_ps(a_21, row_1, c_row_2);
	c_row_2 = _mm_fmadd_ps(a_22, row_2, c_row_2);
	_mm_store_ps(&out.m[6], c_row_2); // This store operation will overwrite the junk that is currently in m[6], and place new junk in m[9] (which is unused)

}

void multiply(Matrix33& out, const Matrix33& A, float scalar) {
	__m256 vec = _mm256_load_ps(&A.m[0]);
	__m256 scal = _mm256_broadcast_ss(&scalar);
	__m256 multiplied = _mm256_mul_ps(vec, scal);
	_mm256_store_ps(&out.m[0], multiplied);
	out.m[8] = A.m[8] * scalar;

}

void transpose(Matrix33& out, Matrix33& in) {
	__m128 row_0 = _mm_load_ps(&in.m[0]); // a0 a1 a2 a3
	__m128 row_1 = _mm_load_ps(&in.m[1]); // a1 a2 a3 a4
	__m128 row_2 = _mm_load_ps(&in.m[2]); // a2 a3 a4 a5
	__m128 row_3 = _mm_load_ps(&in.m[6]); // a6 a7 a8 a9

	// a0   a1   a2   a3       a0   a3   a6   (junk)
	// a3   a4   a5   a6  -->  a1   a4   a7   (junk)
	// a6   a7   a8   a9  -->  a2   a5   a8   (junk)

	_mm_store_ps(&out.m[0], _mm_shuffle_ps(row_0, row_3, 0b00001100));  // returns [a0 a3 a6 a6]
	_mm_store_ps(&out.m[3], _mm_shuffle_ps(row_1, row_3, 0b01011100));  // returns [a1 a4 a7 a7]
	_mm_store_ps(&out.m[6], _mm_shuffle_ps(row_2, row_3, 0b10101100));  // returns [a2 a5 a8 a8]
}


void inverse(Matrix33& out, const Matrix33& A) {
	out.m[0] = A.m[4] * A.m[8] - A.m[5] * A.m[7]; // ei - fh 
	out.m[3] = A.m[5] * A.m[6] - A.m[3] * A.m[8]; // -(di - fg)
	out.m[6] = A.m[3] * A.m[7] - A.m[4] * A.m[6]; // dh - eg
	float determinant = A.m[0] * out.m[0] - A.m[1] * out.m[3] + A.m[2] * out.m[6];
	if (determinant != 0) {
		out.m[1] = A.m[7] * A.m[2] - A.m[1] * A.m[8]; // -(bi - ch)
		out.m[2] = A.m[1] * A.m[5] - A.m[4] * A.m[2]; // bf - ce
		out.m[4] = A.m[0] * A.m[8] - A.m[2] * A.m[6]; // ai - cg
		out.m[5] = A.m[3] * A.m[2] - A.m[0] * A.m[5]; // -(af - cd)
		out.m[7] = A.m[1] * A.m[6] - A.m[0] * A.m[7]; // -(ah - bg)
		out.m[8] = A.m[0] * A.m[4] - A.m[1] * A.m[3]; // ae - bd
		determinant = 1.0f / determinant;
		multiply(out, out, determinant);
	}

}

float trace(const Matrix33& A) {
	return A.m[0] + A.m[4] + A.m[8];
}

void print(const Matrix33& A) {
	std::cout << A.m[0] << ", " << A.m[1] << ", " << A.m[2] << std::endl;
	std::cout << A.m[3] << ", " << A.m[4] << ", " << A.m[5] << std::endl;
	std::cout << A.m[6] << ", " << A.m[7] << ", " << A.m[8] << std::endl;
}


void inverse(Matrix44& out, const Matrix44& A) {

	__m128 row0 = _mm_load_ps(&A.m[0]);  // [a00, a01, a02, a03]
	__m128 row1 = _mm_load_ps(&A.m[4]);  // [a10, a11, a12, a13]
	__m128 row2 = _mm_load_ps(&A.m[8]);  // [a20, a21, a22, a23]
	__m128 row3 = _mm_load_ps(&A.m[12]); // [a30, a31, a32, a33]

	// 2x2 SUBMATRIX CREATION
	__m128 sub_matrix_A = _mm_shuffle_ps(row0, row1, 0b01000100); // [a00, a10, a10, a11]
	__m128 sub_matrix_B = _mm_shuffle_ps(row0, row1, 0b11101110); // [a02, a03, a12, a13]
	__m128 sub_matrix_C = _mm_shuffle_ps(row2, row3, 0b01000100); // [a20, a21, a30, a31]
	__m128 sub_matrix_D = _mm_shuffle_ps(row2, row3, 0b11101110); // [a22, a23, a32, a33]

	// DETERMINANT CALCULATION FOR 2x2 SUBMATRICES
	__m128 det_A = determinant_2x2(sub_matrix_A);
	__m128 det_B = determinant_2x2(sub_matrix_B);
	__m128 det_C = determinant_2x2(sub_matrix_C);
	__m128 det_D = determinant_2x2(sub_matrix_D);

	// ADJUGATE TIMES MATRIX
	// |B|*C - D*(A#B)#  AND |A|D - C(A#B)
	__m128 A_adj_B = adjugate_times_matrix(sub_matrix_A, sub_matrix_B);

	// [dA*c0 dA*c1 dA*c2 dA*c3] - [p0 p1, p2, p3]
	__m128 partial_inverse_B = _mm_sub_ps(_mm_mul_ps(det_B, sub_matrix_C), matrix_times_adjugate(sub_matrix_D, A_adj_B));
	__m128 partial_inverse_D = _mm_sub_ps(_mm_mul_ps(det_A, sub_matrix_D), matrix_times_matrix_2x2(sub_matrix_C, A_adj_B));

	// |D|*A - B*(D#C)  AND |C|B - A(D#C)#
	__m128 D_adj_C = adjugate_times_matrix(sub_matrix_D, sub_matrix_C);

	__m128 partial_inverse_A = _mm_sub_ps(_mm_mul_ps(det_D, sub_matrix_A), matrix_times_matrix_2x2(sub_matrix_B, D_adj_C));
	__m128 partial_inverse_C = _mm_sub_ps(_mm_mul_ps(det_C, sub_matrix_B), matrix_times_adjugate(sub_matrix_A, D_adj_C));
	
	// DETERMINANT CALCULATION 4x4
	__m128 determinant = _mm_add_ps(_mm_mul_ps(det_A, det_D), _mm_mul_ps(det_B, det_C));

	// A#B is ab0 ab1 ab2 ab3
	// C#D is cd0 cd1 cd2 cd3
	//
	// A#B * C#D = ab0  ab1  *  cd0  cd1  =  ab0*cd0+ab1*cd2  ab0*cd1+ab1*cd3  
	//             ab2  ab3     cd2  cd3     ab2*cd0+ab3*cd2  ab2*cd1+ab3*cd3  
	//
	// Tr (A#B * C#D) = ab0*cd0 + ab1*cd2 + ab2*cd1 + ab3*cd3 <-- trABCD
	// 
	// [ab0 ab1 ab2 ab3]
	// [cd0 cd2 cd1 cd3]
	// Mask for 0 2 1 3 -> 0b11011000
	// trABCD -> [ab0*cd0, ab1*cd2, ab2*cd1, ab3*cd3]
	__m128 trABCD = _mm_mul_ps(A_adj_B, _mm_shuffle_ps(D_adj_C, D_adj_C, 0b11011000));

	// trABCD -> [ab0*cd0+ab2*cd1, ab1*cd2+ab3*cd3, ab0*cd0+ab2*cd1, ab1*cd2+ab3*cd3]
	// trABCD -> [ab0*cd0 + ab2*cd1 + ab1*cd2 + ab3*cd3, ditto, ditto, ditto]
	trABCD = _mm_hadd_ps(trABCD, trABCD);
	trABCD = _mm_hadd_ps(trABCD, trABCD);

	determinant = _mm_sub_ps(determinant, trABCD);


	// RECONSTRUCT MATRIX TO GET RESULT
	const __m128 adjugate_sign_mask = _mm_setr_ps(1.0f, -1.0f, -1.0f, 1.0f);
	__m128 reciprocal_determinant = _mm_div_ps(adjugate_sign_mask, determinant);

	partial_inverse_A = _mm_mul_ps(partial_inverse_A, reciprocal_determinant);
	partial_inverse_B = _mm_mul_ps(partial_inverse_B, reciprocal_determinant);
	partial_inverse_C = _mm_mul_ps(partial_inverse_C, reciprocal_determinant);
	partial_inverse_D = _mm_mul_ps(partial_inverse_D, reciprocal_determinant);

	// [a0, a1, b0, b1]       [a3, a1, b3, b1]
	// [a2, a3, b2, b3]  -->  [a2, a0, b2, b0]
	// [c0, c1, d0, d1]  -->  [c3, c1, d3, d1]
	// [c2, c3, d2, d3]       [c2, c0, d2, d0]

	_mm_store_ps(&out.m[0], _mm_shuffle_ps(partial_inverse_A, partial_inverse_B, 0b01110111));
	_mm_store_ps(&out.m[4], _mm_shuffle_ps(partial_inverse_A, partial_inverse_B, 0b00100010));
	_mm_store_ps(&out.m[8], _mm_shuffle_ps(partial_inverse_C, partial_inverse_D, 0b01110111));
	_mm_store_ps(&out.m[12], _mm_shuffle_ps(partial_inverse_C, partial_inverse_D, 0b00100010));
}

void transpose(Matrix44& out, const Matrix44& A) {
	__m128 row_0 = _mm_load_ps(&A.m[0]);  // a0   a1   a2   a3
	__m128 row_1 = _mm_load_ps(&A.m[4]);  // a4   a5   a6   a7
	__m128 row_2 = _mm_load_ps(&A.m[8]);  // a8   a9   a10  a11
	__m128 row_3 = _mm_load_ps(&A.m[12]); // a12  a13  a14  a15

	// a0   a1   a2   a3        a0   a4   a8   a12
	// a4   a5   a6   a7   -->  a1   a5   a9   a13
	// a8   a9   a10  a11  -->  a2   a6   a10  a14
	// a12  a13  a14  a15       a3   a7   a11  a15

	// _mm_unpacklo_ps([a0   a1   a2   a3], [a4   a5   a6   a7]) returns  [a0  a4  a1 a5]
	__m128 row01_helper = _mm_unpacklo_ps(row_0, row_1); // returns [a0  a4  a1 a5]

	// _mm_unpacklo_ps([a8   a9   a10  a11], [a12  a13  a14  a15]) returns [a8  a12  a9  a13]
	__m128 row23_helper = _mm_unpacklo_ps(row_2, row_3); // returns [a8  a12  a9  a13]

	// row0 = _mm_shuffle_ps([a0  a4  a1 a5], [a8  a12  a9  a13]) returns [a0 a4 a8 a12]
	_mm_store_ps(&out.m[0], _mm_shuffle_ps(row01_helper, row23_helper, 0b01000100));

	// row1 = _mm_shuffle_ps([a0  a4  a1 a5], [a8  a12  a9  a13]) returns [a1 a5 a9 a13]
	_mm_store_ps(&out.m[4], _mm_shuffle_ps(row01_helper, row23_helper, 0b11101110));

	// _mm_unpacklo_ps([a0   a1   a2   a3], [a4   a5   a6   a7]) returns  [a2  a6  a3  a7]
	row01_helper = _mm_unpackhi_ps(row_0, row_1); // returns [a2  a6  a3  a7]

	// _mm_unpacklo_ps([a8   a9   a10  a11], [a12  a13  a14  a15]) returns [a10  a14  a11  a15]
	row23_helper = _mm_unpackhi_ps(row_2, row_3); // returns [a10  a14  a11  a15]

	// row2 = _mm_shuffle_ps([a2  a6  a3  a7],  [a10  a14  a11  a15]) returns [a2 a6 a10 a14]
	_mm_store_ps(&out.m[8], _mm_shuffle_ps(row01_helper, row23_helper, 0b01000100));

	// row3 = _mm_shuffle_ps([a2  a6  a3  a7], [a10  a14  a11  a15]) returns [a3 a7 a11 a15]
	_mm_store_ps(&out.m[12], _mm_shuffle_ps(row01_helper, row23_helper, 0b11101110));
}

void multiply(Matrix44& out, const Matrix44& A, const Matrix44& B) {
	__m128 row_0 = _mm_load_ps(&B.m[0]);
	__m128 row_1 = _mm_load_ps(&B.m[4]);
	__m128 row_2 = _mm_load_ps(&B.m[8]);
	__m128 row_3 = _mm_load_ps(&B.m[12]);

	// OUTPUT ROW 0
	__m128 a_00 = _mm_broadcast_ss(&A.m[0]);
	__m128 a_01 = _mm_broadcast_ss(&A.m[1]);
	__m128 a_02 = _mm_broadcast_ss(&A.m[2]);
	__m128 a_03 = _mm_broadcast_ss(&A.m[3]);

	__m128 out_row = _mm_mul_ps(a_00, row_0); // c_row_0 = [a00*b00, a00*b01, a00*b02, a00*b03] 
	out_row = _mm_fmadd_ps(a_01, row_1, out_row); // c_row_0 = [a00*b00, a00*b01, a00*b02, a00*b03] + [a01*b10, a01*b11, a01*b12, a01*b13] 
	out_row = _mm_fmadd_ps(a_02, row_2, out_row); // c_row_0 = [a00*b00+a01*b10, a00*b01+a01*b11, a00*b02+a01*b12, a00*b03+a01*b13] + [a02*b20, a02*b21, a02*b22, a02*b23] 
	out_row = _mm_fmadd_ps(a_03, row_3, out_row);

	_mm_store_ps(&out.m[0], out_row);

	// OUTPUT ROW 1
	__m128 a_10 = _mm_broadcast_ss(&A.m[4]);
	__m128 a_11 = _mm_broadcast_ss(&A.m[5]);
	__m128 a_12 = _mm_broadcast_ss(&A.m[6]);
	__m128 a_13 = _mm_broadcast_ss(&A.m[7]);

	out_row = _mm_mul_ps(a_10, row_0);
	out_row = _mm_fmadd_ps(a_11, row_1, out_row);
	out_row = _mm_fmadd_ps(a_12, row_2, out_row);
	out_row = _mm_fmadd_ps(a_13, row_3, out_row);

	_mm_store_ps(&out.m[4], out_row);

	// OUTPUT ROW 2
	__m128 a_20 = _mm_broadcast_ss(&A.m[8]);
	__m128 a_21 = _mm_broadcast_ss(&A.m[9]);
	__m128 a_22 = _mm_broadcast_ss(&A.m[10]);
	__m128 a_23 = _mm_broadcast_ss(&A.m[11]);

	out_row = _mm_mul_ps(a_20, row_0);
	out_row = _mm_fmadd_ps(a_21, row_1, out_row);
	out_row = _mm_fmadd_ps(a_22, row_2, out_row);
	out_row = _mm_fmadd_ps(a_23, row_3, out_row);
	_mm_store_ps(&out.m[8], out_row);

	// OUTPUT ROW 3
	__m128 a_30 = _mm_broadcast_ss(&A.m[12]);
	__m128 a_31 = _mm_broadcast_ss(&A.m[13]);
	__m128 a_32 = _mm_broadcast_ss(&A.m[14]);
	__m128 a_33 = _mm_broadcast_ss(&A.m[15]);

	out_row = _mm_mul_ps(a_30, row_0);
	out_row = _mm_fmadd_ps(a_31, row_1, out_row);
	out_row = _mm_fmadd_ps(a_32, row_2, out_row);
	out_row = _mm_fmadd_ps(a_33, row_3, out_row);
	_mm_store_ps(&out.m[12], out_row);
}

void multiply(Vector4& out, const Matrix44& A, const Vector4& x) {
	__m128 row_0 = _mm_load_ps(&A.m[0]);  // [a0,  a1,  a2,  a3]
	__m128 row_1 = _mm_load_ps(&A.m[4]);  // [a4,  a5,  a6,  a7]
	__m128 row_2 = _mm_load_ps(&A.m[8]);  // [a8,  a9,  a10, a11]
	__m128 row_3 = _mm_load_ps(&A.m[12]); // [a12, a13, a14, a15]

	__m128 row01_helper = _mm_unpacklo_ps(row_0, row_1);
	__m128 row23_helper = _mm_unpacklo_ps(row_2, row_3);

	__m128 col_0 = _mm_shuffle_ps(row01_helper, row23_helper, 0b01000100);
	__m128 col_1 = _mm_shuffle_ps(row01_helper, row23_helper, 0b11101110);

	row01_helper = _mm_unpackhi_ps(row_0, row_1);
	row23_helper = _mm_unpackhi_ps(row_2, row_3);

	__m128 col_2 = _mm_shuffle_ps(row01_helper, row23_helper, 0b01000100);
	__m128 col_3 = _mm_shuffle_ps(row01_helper, row23_helper, 0b11101110);

	// out0 = [a0*x0 + a1*x1 + a2*x2 + a3*x3]
	// out1 = [a4*x0 + a5*x1 + a6*x2 + a7*x3]
	// out2 = [a8*x0 + a9*x1 + a10*x2 + a11*x3]
	// out3 = [a12*x0 a13*x1 a14*x2 a15*x3]

	// [a0*x0 a1*x1, a2*x2, a3*x3]

	// out_0 = [x0*a0, x0*a4, x0*a8, x0*a12]
	__m128 out_vec = _mm_mul_ps(_mm_broadcast_ss(&x.x), col_0);

	// broadcast -> [x1, x1, x1, x1]
	// [x1, x1, x1, x1] * [a1,  a5,  a9,  a13] = [x1*a1, x1*a5, x1*a9, x1*a13]
	out_vec = _mm_fmadd_ps(_mm_broadcast_ss(&x.y), col_1, out_vec);
	out_vec = _mm_fmadd_ps(_mm_broadcast_ss(&x.z), col_2, out_vec);
	out_vec = _mm_fmadd_ps(_mm_broadcast_ss(&x.w), col_3, out_vec);

	_mm_store_ps((float*) & out, out_vec);
}

inline __m128 adjugate_times_matrix(__m128 vec1, __m128 vec2) {
	//  AB = A# * B
	// If A = a0 a1, then:  A# = a3 -a1, and if B = b0 b1, then: A# * B = a3 -a1  *  b0 b1  =  a3*b0 - a1*b2  a3*b1 - a1*b3
	//        a2 a3              -a2 a0             b2 b3                 -a2 a0     b2 b3     a0*b2 - a2*b0  a0*b3 - a2*b1
	//
	// The calculation of the resulting matrix can be expanded as:
	// 
	// [a3*b0 - a1*b2, a3*b1 - a1*b3, a0*b2 - a2*b0, a0*b3 - a2*b1]
	// [a3*b0, a3*b1, a0*b2, a0*b3] - [a1*b2, a1*b3, a2*b0, a2*b1]
	// [a3, a3, a0, a0] * [b0, b1, b2, b3] - [a1, a1, a2, a2] * [b2, b3, b0, b1]

	// Thus, for 2x2 matrices a = [a0, a1, a2, a3], and b = [b0, b1, b2, b3]
	// _mm_shuffle_ps([a0, a1, a2, a3], [a0, a1, a2, a3], 0b00001111) : returns [a3, a3, a0, a0]
	// _mm_mul_ps([a3, a3, a0, a0], [b0, b1, b2, b3]) : returns [a3*b0, a3*b1, a0*b2, a0*b3] <- load into mat1
	// _mm_shuffle_ps([a0, a1, a2, a3], [a0, a1, a2, a3], 0b10100101) : returns [a1, a1, a2, a2]
	// _mm_shuffle_ps([b0, b1, b2, b3], [b0, b1, b2, b3], 0b01001110) : returns [b2, b3, b0, b1]
	// _mm_mul_ps([a1, a1, a2, a2], [b2, b3, b0, b1]) : returns [a1*b2, a1*b3, a2*b0, a2*b1] <- load into mat2
	// _mm_sub_ps([a3*b0, a3*b1, a0*b2, a0*b3], [a1*b2, a1*b3, a2*b0, a2*b1]) : returns [a3*b0 - a1*b2, a3*b1 - a1*b3, a0*b2 - a2*b0, a0*b3 - a2*b1]
	__m128 mat1 = _mm_mul_ps(_mm_shuffle_ps(vec1, vec1, 0b00001111), vec2);
	__m128 mat2 = _mm_mul_ps(_mm_shuffle_ps(vec1, vec1, 0b10100101), _mm_shuffle_ps(vec2, vec2, 0b01001110));
	return _mm_sub_ps(mat1, mat2);
}

inline __m128 matrix_times_adjugate(__m128 vec1, __m128 vec2) {
	// If A = a0 a1 and B b0 b1, with B# =  b3 -b1. Then A * B# = a0 a1  *   b3 -b1  =  a0*b3 - a1*b2  a1*b0 - a0*b1
	//        a2 a3       b2 b3            -b2  b0                a2 a3  *  -b2  b0     a2*b3 - a3*b2  a3*b0 - a2*b1
	//
	// The calculation of the resulting matrix can be expanded as:
	// 
	// [a0*b3 - a1*b2, a1*b0 - a0*b1, a2*b3 - a3*b2, a3*b0 - a2*b1]
	// [a0*b3, a1*b0, a2*b3, a3*b0] - [a1*b2, a0*b1, a3*b2, a2*b1]
	// [a0, a1, a2, a3] * [b3, b0, b3, b0] - [a1, a0, a3, a2] * [b2, b1, b2, b1]
	// 
	// _mm_shuffle_ps([b0, b1, b2, b3], [b0, b1, b2, b3], 0b11001100) : returns [b3, b0, b3, b0]
	// _mm_mul_ps([a0, a1, a2, a3], [b3, b0, b3, b0]) : returns [a0*b3, a1*b0, a2*b3, a3*b0] <- load into mat1
	// _mm_shuffle_ps([a0, a1, a2, a3], [a0, a1, a2, a3], 0b10110001) : returns [a1, a0, a3, a2]
	// _mm_shuffle_ps([b0, b1, b2, b3], [b0, b1, b2, b3], 0b01100110) : returns [b2, b1, b2, b1]
	// _mm_mul_ps([a1, a0, a3, a2], [b2, b1, b2, b1]) : returns [a1*b2, a0*b1, a3*b2, a2*b1] <- load into mat2
	// _mm_sub_ps([a0*b3, a1*b0, a2*b3, a3*b0], [a1*b2, a0*b1, a3*b2, a2*b1]) : returns [a0*b3 - a1*b2, a1*b0 - a0*b1, a2*b3 - a3*b2, a3*b0 - a2*b1]
	__m128 mat1 = _mm_mul_ps(vec1, _mm_shuffle_ps(vec2, vec2, 0b00110011));
	__m128 mat2 = _mm_mul_ps(_mm_shuffle_ps(vec1, vec1, 0b10110001), _mm_shuffle_ps(vec2, vec2, 0b01100110));
	return _mm_sub_ps(mat1, mat2);
}

inline __m128 matrix_times_matrix_2x2(__m128 vec1, __m128 vec2) {
	// a0 a1   b0 b1   a0*b0 + a1*b2  a0*b1 + a1*b3
	// a2 a3 * b2 b3 = a2*b0 + a3*b2  a2*b1 + a3*b3
	//
	// [a0*b0 + a1*b2, a0*b1 + a1*b3, a2*b0 + a3*b2, a2*b1 + a3*b3]
	// [a0*b0, a0*b1, a2*b0, a2*b1] + [a1*b2, a1*b3, a3*b2, a3*b3]
	// [a0 a0 a2 a2] * [b0 b1 b0 b1]  +  [a1 a1 a3 a3] * [b2 b3 b2 b3]
	// 
	// _mm_shuffle_ps([a0, a1, a2, a3], [a0, a1, a2, a3], 0b10100000) : returns [a0 a0 a2 a2]
	// _mm_shuffle_ps([b0, b1, b2, b3], [b0, b1, b2, b3], 0b01000100) : returns [b0 b1 b0 b1]
	// _mm_mul_ps([a0 a0 a2 a2], [b0 b1 b0 b1]) : returns [a0*b0, a0* b1, a2*b0, a2*b1] <- load into mat1
	// _mm_shuffle_ps([a0, a1, a2, a3], [a0, a1, a2, a3], 0b11110101) : returns [a1 a1 a3 a3]
	// _mm_shuffle_ps([b0, b1, b2, b3], [b0, b1, b2, b3], 0b11101110) : returns [b2 b3 b2 b3]
	// _mm_mul_ps([a1 a1 a3 a3], [b2 b3 b2 b3]) : returns [a1*b2 a1*b3 a3*b2 a3*b3] <- load into mat2
	// _mm_add_ps([a0*b0, a0* b1, a2*b0, a2*b1], [a1*b2 a1*b3 a3*b2 a3*b3]) : returns [a0*b0 + a1*b2, a0*b1 + a1*b3, a2*b0 + a3*b2, a2*b1 + a3*b3]
	__m128 mat1 = _mm_mul_ps(_mm_shuffle_ps(vec1 ,vec1, 0b10100000), _mm_shuffle_ps(vec2, vec2, 0b01000100));
	__m128 mat2 = _mm_mul_ps(_mm_shuffle_ps(vec1, vec1, 0b11110101), _mm_shuffle_ps(vec2, vec2, 0b11101110));
	return _mm_add_ps(mat1, mat2);
}

inline __m128 determinant_2x2(__m128 matrix) {
	// with M = m0 m1   det(M) = m0*m3 - m1*m2
	//          m2 m3
	// 
	// for 2x2 matrix m = [m0, m1, m2, m3]
	// _mm_shuffle_ps([m0, m1, m2, m3], [m0, m1, m2, m3], 0b00011011) : returns [m3 m2 m1 m0]
	// _mm_mul_ps([m3, m2, m1, m0], [m0, m1, m2, m3]) : returns [m3*m0, m2*m1, m1*m2, m0*m3] <- load into det_A
	// _mm_shuffle_ps(det_A, det_A, 0b11010101) : return [m2*m1, m2*m1, m2*m1, m0*m3]
	// _mm_sub_ps([m3*m0, m2*m1, m1*m2, m0*m3], [m2*m1, m2*m1, m2*m1, m0*m3]) : returns [m3*m0 - m2*m1, 0, 0, 0] <- load into det_A
	// _mm_shuffle_ps(det_A, det_A, 0) : returns [m3*m0 - m2*m1, m3*m0 - m2*m1, m3*m0 - m2*m1, m3*m0 - m2*m1] <- load into det_A
	__m128 det = _mm_mul_ps(_mm_shuffle_ps(matrix, matrix, 0b00011011), matrix);
	det = _mm_sub_ps(det, _mm_shuffle_ps(det, det, 0b11010101));
	return _mm_shuffle_ps(det, det, 0);
}

void print(Matrix44& matrix) {
	std::cout << std::endl;
	std::cout << matrix.m[0] << ", " << matrix.m[1] << ", " << matrix.m[2] << ", " << matrix.m[3] << std::endl;
	std::cout << matrix.m[4] << ", " << matrix.m[5] << ", " << matrix.m[6] << ", " << matrix.m[7] << std::endl;
	std::cout << matrix.m[8] << ", " << matrix.m[9] << ", " << matrix.m[10] << ", " << matrix.m[11] << std::endl;
	std::cout << matrix.m[12] << ", " << matrix.m[13] << ", " << matrix.m[14] << ", " << matrix.m[15] << std::endl;
	std::cout << std::endl;
}

float dot(const Vector4& A, const Vector4& B) {
	return A.x * B.x + A.y * B.y + A.z * B.z + A.w * B.w;
}

void dot_batch(float* out, const Vector4& A, Vector4* vectors, int num_vectors) {
	// first batch
	// __m128 can hold 1 vector

	// vec_a        = [xa, ya, za, wa]
	// vecs         = [x1, y1, z1, w1]
	__m128 vec_a = _mm_load_ps((float*)&A);
	
	for (int i = 0; i < num_vectors; i++) {
		__m128 vec = _mm_load_ps((float*)&vectors[i]);

		// r1 = [xa*x1, ya*y1, za*z1, wa*w1];
		__m128 r1 = _mm_mul_ps(vec_a, vec);

		// [ya*y1, xa*x1, wa*w1, za*z1]
		__m128 shuf = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(2, 3, 0, 1));

		// [xa*x1, ya*y1, za*z1, wa*w1] + [ya*y1, xa*x1, wa*w1, za*z1]
		// [xa*x1+ya*y1, ya*y1+xa*x1, za*z1+wa*w1, wa*w1+za*z1]
		__m128 sums = _mm_add_ps(r1, shuf);

		// [sums2, sums3, shuf2 shuf3]
		// [wa*w1+ya*y1, za*z1+xa*x1, xa*x1, ya*y1]
		shuf = _mm_movehl_ps(shuf, sums);

		// [xa*x1+za*z1, ya*y1+wa*w1, za*z1+xa*x1, wa*w1+ya*y1] + [za*z1+xa*x1, wa*w1+ya*y1, xa*x1, ya*y1]
		// [xa*x1+za*z1+ya*y1+wa*w1, ya*y1+wa*w1, za*z1+xa*x1, wa*w1+ya*y1]
		sums = _mm_add_ss(sums, shuf);

		out[i] = _mm_cvtss_f32(sums);
	}
}