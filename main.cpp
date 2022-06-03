#include <cmath>
#include <iostream>
#include <string>

#include "Config.h"
#include "benchmark_tests.h"


int main() {
	BENCHMARK_MATRIX_MULTIPLICATION();
	BENCHMARK_MATRIX_INVERSE();
	BENCHMARK_MATRIX_SCALAR_MULT();
	BENCHMARK_MATRIX_TRANSPOSE();
	BENCHMARK_VECTOR_DOT();
	return 1;
}