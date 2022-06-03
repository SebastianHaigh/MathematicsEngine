#include <iostream>

#include "benchmarking.h"

Timer::Timer() {
	start_timepoint = std::chrono::high_resolution_clock::now();
}

Timer::~Timer() {
	stop();
}

void Timer::stop() {
	auto end_timepoint = std::chrono::high_resolution_clock::now();
	auto start = std::chrono::time_point_cast<std::chrono::microseconds>(start_timepoint).time_since_epoch().count();
	auto stop = std::chrono::time_point_cast<std::chrono::microseconds>(end_timepoint).time_since_epoch().count();

	auto duration = stop - start;

	std::cout << duration << " us" << std::endl;
}