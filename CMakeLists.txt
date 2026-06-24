cmake_minimum_required(VERSION 3.16)
project (NQS LANGUAGES CXX)

file(GLOB_RECURSE SOURCES
    "${PROJECT_SOURCE_DIR}/src/main.cc"
    "${PROJECT_SOURCE_DIR}/src/MolLoader.cc"
    "${PROJECT_SOURCE_DIR}/src/Hamiltonian.cc"
    "${PROJECT_SOURCE_DIR}/src/Energy.cc"
    "${PROJECT_SOURCE_DIR}/src/RBM.cc"
    "${PROJECT_SOURCE_DIR}/src/mcmc.cc"
    "${PROJECT_SOURCE_DIR}/src/SROpt.cc"
    "${PROJECT_SOURCE_DIR}/src/utils/*.cc"
)

include_directories("${PROJECT_SOURCE_DIR}/src")

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

include_directories(
    "${PROJECT_SOURCE_DIR}/library"
)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE pybind11::embed)

set(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -O2 -Wall -g -march=native -ggdb -fopenmp -std=c++17")
