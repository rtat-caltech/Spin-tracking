message("Processing tests/CMakeLists.txt")

if (CMAKE_CUDA_ARCHITECTURES)
  set_source_files_properties(test_main.cpp ${UNIT_TESTS} PROPERTIES LANGUAGE CUDA)
endif ()
add_executable(test test_main.cpp ${UNIT_TESTS})
target_link_libraries(test mainlib)
