#include <chrono>

class Timer {
private:
	std::chrono::time_point<std::chrono::high_resolution_clock> start_timepoint;

public:
	Timer();
	~Timer();
	void stop();
};