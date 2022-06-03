# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.10)

project(MEngine VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Get GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

configure_file(Config.h.in Config.h)


add_subdirectory (MathematicsEngine)
list(APPEND EXTRA_LIBS MathematicsEngine)

add_subdirectory (Benchmarking)
list(APPEND EXTRA_LIBS Benchmarking)

# add the executable
add_executable(MEngine "main.cpp" "Benchmarking/timer.cpp" "Benchmarking/benchmarking.h" "benchmark_tests.cpp" "benchmark_tests.h")

target_link_libraries(MEngine PUBLIC ${EXTRA_LIBS})

target_include_directories(MEngine PUBLIC "${PROJECT_BINARY_DIR}")


install(TARGETS MEngine DESTINATION bin)
install(FILES "${PROJECT_BINARY_DIR}/Config.h" DESTINATION include)

# TESTING

enable_testing()

add_executable(MathematicsTest ./Test/mathematics_test.cpp ./MathematicsEngine/MatrixAndVector.cpp)
target_link_libraries(MathematicsTest gtest_main)

include(GoogleTest)
gtest_discover_tests(MathematicsTest)