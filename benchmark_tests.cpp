#include "benchmark_tests.h"
#include "MathematicsEngine.h"

void BENCHMARK_MATRIX_MULTIPLICATION() {
	std::cout << std::endl;
	std::cout << "-----------------------" << std::endl;
	std::cout << "BENCHMARK_MATRIX_MULTIPLICATION" << std::endl;
	std::cout << "-----------------------" << std::endl;

	std::cout << std::endl << "Time for Matrix33_SIMD: " << std::endl;
	{
		Matrix33 A(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
		Matrix33 B(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
		Matrix33 C;
		Timer timer;
		for (int i = 0; i < 1000000; i++) {
			multiply(C, A, B);
		}
	}

	std::cout << std::endl << "Time for Matrix44_s: " << std::endl;
	{
		Matrix44 A(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
		Matrix44 B(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
		Matrix44 C;
		Timer timer;
		for (int i = 0; i < 1000000; i++) {
			multiply(C, A, B);
		}
	}
	
}

void BENCHMARK_MATRIX_INVERSE() {

	std::cout << std::endl;
	std::cout << "-----------------------" << std::endl;
	std::cout << "BENCHMARK_MATRIX_INVERSE" << std::endl;
	std::cout << "-----------------------" << std::endl;


	std::cout << std::endl << "Time for Matrix33: " << std::endl;
	{
		Matrix33 B(12.0, 2.0, 3.0, 4.0, 16.0, 6.0, 7.0, 8.0, 19.0);
		Matrix33 C;
		Timer timer;
		for (int i = 0; i < 1000000; i++) {
			inverse(C, B);
		}
	}

	std::cout << std::endl << "Time for Matrix44_s: " << std::endl;
	{
		Matrix44 M;
		Matrix44 I;

		M.m[0] = 90.0; M.m[1] = 73.0; M.m[2] = 3.0; M.m[3] = 4.0;
		M.m[4] = 1.0; M.m[5] = 16.0; M.m[6] = 7.0; M.m[7] = 8.0;
		M.m[8] = 1.0; M.m[9] = 3.0; M.m[10] = 19.0; M.m[11] = 81.2;
		M.m[12] = 2.0; M.m[13] = 1.0; M.m[14] = 101.8; M.m[15] = 15.0;

		Timer timer;
		for (int i = 0; i < 1000000; i++) {
			inverse(I, M);
		}
	}
}


void BENCHMARK_MATRIX_SCALAR_MULT() {

	std::cout << std::endl;
	std::cout << "-----------------------" << std::endl;
	std::cout << "BENCHMARK_MATRIX_SCALAR_MULT" << std::endl;
	std::cout << "-----------------------" << std::endl;

	std::cout << "Time for Matrix33_SIMD: " << std::endl;
	{
		Matrix33 A(12.0, 2.0, 3.0, 4.0, 16.0, 6.0, 7.0, 8.0, 19.0);
		Matrix33 B;
		float c = 0.9999999;
		Timer timer;
		for (int i = 0; i < 1000000; i++) {
			multiply(B, A, c);
		}
	}

}

void BENCHMARK_MATRIX_TRANSPOSE() {

	std::cout << std::endl;
	std::cout << "-----------------------" << std::endl;
	std::cout << "BENCHMARK_MATRIX_TRANSPOSE" << std::endl;
	std::cout << "-----------------------" << std::endl;



	std::cout << std::endl << "Time for Matrix33: " << std::endl;
	{
		Matrix33 B(12.0, 2.0, 3.0, 4.0, 16.0, 6.0, 7.0, 8.0, 19.0);
		Matrix33 C;
		Timer timer;
		for (int i = 0; i < 1000000; i++) {
			transpose(C, B);
		}
	}

	std::cout << std::endl << "Time for Matrix44_s: " << std::endl;
	{
		Matrix44 A{ 90.0, 73.0, 3.0, 4.0, 1.0, 16.0, 7.0, 8.0, 1.0, 3.0, 19.0, 81.0, 2.0, 1.0, 101.0, 15.0 };
		Matrix44 C;
		Timer timer;
		for (int i = 0; i < 1000000; i++) {
			transpose(C, A);
		}
	}

}

void BENCHMARK_VECTOR_DOT() {

	std::cout << std::endl;
	std::cout << "-----------------------" << std::endl;
	std::cout << "BENCHMARK_VECTOR_DOT" << std::endl;
	std::cout << "-----------------------" << std::endl;

	std::cout << std::endl << "Time for dot: " << std::endl;
	{
		Vector4 A(12.0, 2.0, 3.0, 4.0);
		Vector4 B(9.0, 12.0, 7.0, 8.0);
		Timer timer;
		for (int i = 0; i < 50000000; i++) {
			float d = dot(A, B);
		}
	}

	std::cout << std::endl << "Time for dot batch: " << std::endl;
	{
		Vector4 A(12.0, 2.0, 3.0, 4.0);
		Vector4 B[50];
		float d[50] = { 0.0f };
		for (int i = 0; i < 50; i++) {
			B[i] = Vector4{ 1.0f, 2.0f, 3.0f, (float)i };
		}
		Timer timer;
		for (int i = 0; i < 1000000; i++) {
			dot_batch(d, A, B, 50);
		}
	}

}