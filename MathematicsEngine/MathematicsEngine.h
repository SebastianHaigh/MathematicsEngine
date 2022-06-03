#ifndef MATHEMATICS_ENGINE_H_
#define MATHEMATICS_ENGINE_H_

#include <iostream>
#include <immintrin.h>

struct alignas(64) Matrix33 {
	float m[10];
	Matrix33() : m{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 } {}
	Matrix33(float x) : m{ x, x, x, x, x, x, x, x, x, 0.0 } {}
	Matrix33(float m00, float m01, float m02,
		float m10, float m11, float m12,
		float m20, float m21, float m22)
		: m{ m00, m01, m02, m10, m11, m12, m20, m21, m22, 0.0 } {}
};

struct alignas(64) Matrix44 {
	float m[16];
	Matrix44() : m{ 0.0f } {}
	Matrix44(float m0, float m1, float m2, float m3,
		float m4, float m5, float m6, float m7,
		float m8, float m9, float m10, float m11,
		float m12, float m13, float m14, float m15)
		:m{ m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15 } {}
};

struct alignas(16) Vector4 {
	float x, y, z, w;
	Vector4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
	Vector4(float x, float y, float z, float w)
		: x(x), y(y), z(z), w(w) {}
};

void add(Matrix33& out, const Matrix33& A, const Matrix33& B);
void transpose(Matrix33& out, Matrix33& in);
void multiply(Matrix33& out, const Matrix33& A, const Matrix33& B);
void multiply(Matrix33& out, const Matrix33& A, float scalar);
void inverse(Matrix33& out, const Matrix33& A);
void print(Matrix33& A);

void transpose(Matrix44& out, const Matrix44& A);
void inverse(Matrix44& out, const Matrix44& A);
void multiply(Matrix44& out, const Matrix44& A, const Matrix44& B);
void multiply(Vector4& out, const Matrix44& A, const Vector4& x);
void print(const Matrix44& matrix);

float dot(const Vector4& A, const Vector4& B);
void dot_batch(float* out, const Vector4& A, Vector4* vectors, int num_vectors);


__m128 matrix_times_matrix_2x2(__m128 vec1, __m128 vec2);
__m128 determinant_2x2(__m128 matrix);
__m128 adjugate_times_matrix(__m128 vec1, __m128 vec2);
__m128 matrix_times_adjugate(__m128 vec1, __m128 vec2);

#endif // MATHEMATICS_ENGINE_H_