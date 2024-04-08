# CMakeLists.txt

cmake_minimum_required(VERSION 3.12)

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

# Unit Testing
add_subdirectory(tests)
target_compile_options(test PRIVATE -w)

find_package(HDF5 COMPONENTS CXX)
if (HDF5_FOUND)
  include_directories(${HDF5_INCLUDE_DIRS})
  target_compile_definitions(mainlib PUBLIC USEHDF5=1)
  target_link_libraries(mainlib ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})
else()
  message(STATUS "Could not find HDF5 library - HDF5 saving be disabled")
endif()

IF (CMAKE_CUDA_ARCHITECTURES)
  find_package(CUDA 11.8 REQUIRED)
  find_package(CUDAToolkit REQUIRED)
  if(NOT DEFINED ENV{CUTENSOR_ROOT} AND NOT DEFINED CUTENSOR_ROOT)
	message(FATAL_ERROR "CUTENSOR_ROOT not set!")
  else()
	if(DEFINED ENV{CUTENSOR_ROOT})
	  set(CUTENSOR_ROOT "$ENV{CUTENSOR_ROOT}")
	endif()
	message("-- Looking for cuTENSOR in ${CUTENSOR_ROOT}")
	if(NOT EXISTS ${CUTENSOR_ROOT})
	  message(FATAL_ERROR "Cannot find CUTENSOR_ROOT")
	endif()
  endif()

  if(NOT TARGET cutensor)
	add_library(cutensor SHARED IMPORTED)
	if(WIN32)
	  set(CUTENSOR_LIBRARY_NAME "cutensor.dll")
	  set(CUTENSOR_LIBRARY_DEF "cutensor.lib")
	else()
	  set(CUTENSOR_LIBRARY_NAME "libcutensor.so")
	endif()
	if("${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}" VERSION_GREATER_EQUAL "11.1")
	  set(LIB_DIR "/libcutensor/${CUDA_VERSION_MAJOR}")
	else()
	  set(LIB_DIR "/libcutensor/${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}")
	endif()
	set_target_properties(cutensor PROPERTIES
	  IMPORTED_LOCATION "${CUTENSOR_ROOT}/${LIB_DIR}/${CUTENSOR_LIBRARY_NAME}"
	  IMPORTED_IMPLIB "${CUTENSOR_ROOT}/${LIB_DIR}/${CUTENSOR_LIBRARY_DEF}")
  endif()
  # If Cuda Architecture is set (via commandline option -DCMAKE_CUDA_ARCHITECTURES=XX)
  message("  CUDA Arch: '${CMAKE_CUDA_ARCHITECTURES}'")
  message("  CUDA Flags: '${CMAKE_CUDA_FLAGS}'")
  # Normally CMake interprets .cpp extension as C++. This makes it interpret these files as CUDA instead.
  set_source_files_properties(main.cpp ${SOURCES} PROPERTIES LANGUAGE CUDA)
  set_target_properties(mainlib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_include_directories(mainlib
	PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	${CMAKE_SOURCE_DIR}/../utils)
  target_link_libraries(mainlib cutensor CUDA::cufft CUDA::cudart)
  
ENDIF () 

add_executable(main main.cpp)
target_compile_options(main PRIVATE -w) # Suppress warnings
target_link_libraries(main mainlib)
