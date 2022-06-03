add_library(Benchmarking timer.cpp)
target_include_directories(Benchmarking INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS Benchmarking DESTINATION lib)
install(FILES benchmarking.h DESTINATION include)