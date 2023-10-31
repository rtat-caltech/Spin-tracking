message("Processing tests/CMakeLists.txt")

#set(HEADER_DIRECTORY "${CMAKE_SOURCE_DIRECTORY}/include")
#file(GLOB SOURCES "${CMAKE_SOURCE_DIRECTORY}/src/*.cpp")
#file(GLOB LOCAL_HEADERS "${CMAKE_SOURCE_DIRECTORY}/${HEADER_DIRECTORY}/*.h")

add_executable(test test_main.cpp ${UNIT_TESTS})
target_link_libraries(test mainlib)
