add_library(MathematicsEngine "MatrixAndVector.cpp")
target_include_directories(MathematicsEngine INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS MathematicsEngine DESTINATION lib)
install(FILES MathematicsEngine.h DESTINATION include)