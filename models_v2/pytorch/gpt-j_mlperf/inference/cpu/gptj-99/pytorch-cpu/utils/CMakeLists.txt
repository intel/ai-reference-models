cmake_minimum_required(VERSION 3.1)
project(CORE_BIND)

find_package(OpenMP REQUIRED)
find_package(pybind11 REQUIRED)

file (GLOB SOURCE_FILES "./*.cpp")
file (GLOB HEADER_FILES "./*.hpp")


pybind11_add_module(thread_binder ${SOURCE_FILES} ${HEADER_FILES})

target_link_libraries(thread_binder PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(thread_binder PUBLIC OpenMP::OpenMP_CXX)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")