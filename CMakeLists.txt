# CMakeLists.txt

cmake_minimum_required(VERSION 3.8)

project(Spin-tracking LANGUAGES CUDA CXX)

message("  C: '${CMAKE_C_COMPILER}'")
message("  C++: '${CMAKE_CXX_COMPILER}'")

set(HEADER_DIRECTORY include)
set(TEST_DIRECTORY tests)
set(SOURCE_DIRECTORY src)

file(GLOB SOURCES CONFIGURE_DEPENDS "${SOURCE_DIRECTORY}/[a-zA-Z]*.cpp")
file(GLOB UNIT_TESTS CONFIGURE_DEPENDS "${TEST_DIRECTORY}/[a-zA-Z]*.cpp")
file(GLOB LOCAL_HEADERS CONFIGURE_DEPENDS "${HEADER_DIRECTORY}/[a-zA-Z]*.h")


# Create library
add_library(mainlib ${SOURCES} ${LOCAL_HEADERS})
target_include_directories(mainlib PRIVATE ${HEADER_DIRECTORY})

target_compile_features(mainlib PUBLIC cxx_std_17)
target_compile_options(mainlib PRIVATE -w -lineinfo) # Suppress warnings

find_package(HDF5 REQUIRED COMPONENTS CXX)

IF (CMAKE_CUDA_ARCHITECTURES)
  find_package(CUDA 11.8 REQUIRED)
  # If Cuda Architecture is set (via commandline option -DCMAKE_CUDA_ARCHITECTURES=XX)
  message("  CUDA Arch: '${CMAKE_CUDA_ARCHITECTURES}'")
  message("  CUDA Flags: '${CMAKE_CUDA_FLAGS}'")
  # Normally CMake interprets .cpp extension as C++. This makes it interpret these files as CUDA instead.
  set_source_files_properties(main.cpp ${SOURCES}  PROPERTIES LANGUAGE CUDA)
  set_target_properties(mainlib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
ELSE ()
  # Unit Testing Version
  add_subdirectory(tests)
  target_compile_options(test PRIVATE -w)
ENDIF ()

add_executable(main main.cpp)
target_compile_options(main PRIVATE -w) # Suppress warnings
target_link_libraries(mainlib ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})
target_link_libraries(main mainlib)
