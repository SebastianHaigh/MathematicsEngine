#include <gtest/gtest.h>
#include "../MathematicsEngine/MathematicsEngine.h"

// Demonstrate some basic assertions.
TEST(Matrix44Test, BasicAssertions) {
	// Arrange
	Matrix44 M, I;
	M.m[0] = 90.0f; M.m[1] = 73.0f; M.m[2] = 3.0f; M.m[3] = 4.0f;
	M.m[4] = 1.0f; M.m[5] = 16.0f; M.m[6] = 7.0f; M.m[7] = 8.0f;
	M.m[8] = 1.0f; M.m[9] = 3.0f; M.m[10] = 19.0f; M.m[11] = 81.2f;
	M.m[12] = 2.0f; M.m[13] = 1.0f; M.m[14] = 101.8f; M.m[15] = 15.0f;

	// Act
	inverse(I, M);

	// Assert
	EXPECT_NEAR(I.m[0], 0.0116, 0.001);
	EXPECT_NEAR(I.m[1], -0.0539, 0.001);
	EXPECT_NEAR(I.m[2], 0.0043, 0.001);
	EXPECT_NEAR(I.m[3], 0.0026, 0.001);
}

TEST(Matrix44Test, MatrixMultiplication) {
	// Arrange
	Matrix44 A(10, 7, 9, 32, 8, 3, 10, 82, 81, 37, 39, 1, 92, 9, 7, 2);
	Matrix44 B{ 90.0, 73.0, 3.0, 4.0, 1.0, 16.0, 7.0, 8.0, 1.0, 3.0, 19.0, 81.0, 2.0, 1.0, 101.0, 15.0 };
	Matrix44 C;
	Matrix44 expected{ 980.0f, 901.0f, 3482.0f, 1305.0f,
		897.0f, 744.0f, 8517.0f, 2096.0f,
		7368.0f, 6623.0f, 1344.0f, 3794.0f,
		8300.0f, 6883.0f, 674.0f, 1037.0f };

	// Act
	multiply(C, A, B);

	// Assert
	EXPECT_EQ(C.m[0], expected.m[0]);
	EXPECT_EQ(C.m[1], expected.m[1]);
	EXPECT_EQ(C.m[2], expected.m[2]);
	EXPECT_EQ(C.m[3], expected.m[3]);
	EXPECT_EQ(C.m[4], expected.m[4]);
	EXPECT_EQ(C.m[5], expected.m[5]);
	EXPECT_EQ(C.m[6], expected.m[6]);
	EXPECT_EQ(C.m[7], expected.m[7]);
	EXPECT_EQ(C.m[8], expected.m[8]);
	EXPECT_EQ(C.m[9], expected.m[9]);
	EXPECT_EQ(C.m[10], expected.m[10]);
	EXPECT_EQ(C.m[11], expected.m[11]);
	EXPECT_EQ(C.m[12], expected.m[12]);
	EXPECT_EQ(C.m[13], expected.m[13]);
	EXPECT_EQ(C.m[14], expected.m[14]);
	EXPECT_EQ(C.m[15], expected.m[15]);
}

TEST(Matrix44Test, MatrixVectorMultiplication) {
	// Arrange
	Matrix44 A{ 90.0, 73.0, 3.0, 4.0, 1.0, 16.0, 7.0, 8.0, 1.0, 3.0, 19.0, 81.0, 2.0, 1.0, 101.0, 15.0 };
	Vector4 y;
	Vector4 x = { 2.0f, 3.0f, 4.0f, 5.0f };
	Vector4 expected{ 431.0f, 118.0f, 492.0f, 486.0f };

	// Act
	multiply(y, A, x);

	// Assert
	EXPECT_EQ(y.x, expected.x);
	EXPECT_EQ(y.y, expected.y);
	EXPECT_EQ(y.z, expected.z);
	EXPECT_EQ(y.w, expected.w);
}

TEST(Matrix44Test, MatrixTranspose) {
	// Arrange
	Matrix44 A{ 90.0, 73.0, 3.0, 4.0, 1.0, 16.0, 7.0, 8.0, 1.0, 3.0, 19.0, 81.0, 2.0, 1.0, 101.0, 15.0 };
	Matrix44 B;
	Matrix44 expected{ 90.0f, 1.0f, 1.0f, 2.0f, 73.0f, 16.0f, 3.0f, 1.0f, 3.0f, 7.0f, 19.0f, 101.0f, 4.0f, 8.0f, 81.0f, 15.0f };

	// Act
	transpose(B, A);

	// Assert
	EXPECT_EQ(B.m[0], expected.m[0]);
	EXPECT_EQ(B.m[1], expected.m[1]);
	EXPECT_EQ(B.m[2], expected.m[2]);
	EXPECT_EQ(B.m[3], expected.m[3]);
	EXPECT_EQ(B.m[4], expected.m[4]);
	EXPECT_EQ(B.m[5], expected.m[5]);
	EXPECT_EQ(B.m[6], expected.m[6]);
	EXPECT_EQ(B.m[7], expected.m[7]);
	EXPECT_EQ(B.m[8], expected.m[8]);
	EXPECT_EQ(B.m[9], expected.m[9]);
	EXPECT_EQ(B.m[10], expected.m[10]);
	EXPECT_EQ(B.m[11], expected.m[11]);
	EXPECT_EQ(B.m[12], expected.m[12]);
	EXPECT_EQ(B.m[13], expected.m[13]);
	EXPECT_EQ(B.m[14], expected.m[14]);
	EXPECT_EQ(B.m[15], expected.m[15]);
}


TEST(Vector4Test, DotProduct) {
	// Arrange
	Vector4 a{ 1.0f, 2.0f, 3.0f, 4.0f };
	Vector4 b{ 4.0f, 3.0f, 2.0f, 1.0f };
	float expected = 20.0f;

	// Act
	float actual = dot(a, b);

	// Assert
	EXPECT_EQ(expected, actual);
}

TEST(Vector4Test, DotBatchProduct) {
	// Arrange
	Vector4 a{ 4.0f, 3.0f, 2.0f, 1.0f };
	Vector4 b[50];
	float expected[50] = { 0.0f };
	for (int i = 0; i < 50; i++) {
		b[i] = Vector4{ 1.0f, 2.0f, 3.0f, (float)i };
		expected[i] = 16.0f + i;
	}

	// Act
	float actual[50] = { 0.0f };
	dot_batch(&actual[0], a, &b[0], 50);

	// Assert
	for (int i = 0; i < 50; i++) {
		EXPECT_EQ(expected[i], actual[i]);
	}
	
}