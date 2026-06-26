cmake_minimum_required(VERSION 3.10)

# Set project name and language
project(vectormaton CXX)

find_package(OpenMP REQUIRED)

# Enable optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Set C++ standard (C++17 or later is required for modern features)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(VECTORMATON_INCLUDE_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/source"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/baselines"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/hnswlib")

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/hnswlib/hnswlib/hnswlib.h")
    message(FATAL_ERROR
        "hnswlib headers were not found. Run: git submodule update --init --recursive")
endif()

set(VECTORMATON_CORE_SOURCES
    source/headers.h
    source/mpmc_queue.h
    source/sa.h
    source/sa.cpp
    source/vectormaton.h
    source/vectormaton.cpp)

set(VECTORMATON_BASELINE_SOURCES
    source/baselines/exact.h
    source/baselines/exact.cpp
    source/baselines/opt_query.h
    source/baselines/opt_query.cpp
    source/baselines/pre_filtering.h
    source/baselines/pre_filtering.cpp
    source/baselines/post_filtering.h
    source/baselines/post_filtering.cpp
    source/baselines/bm25_filtering.h
    source/baselines/bm25_filtering.cpp
    source/baselines/hybrid.h
    source/baselines/hybrid.cpp)

option(ENABLE_ACORN "Build the ACORN external baseline wrapper." OFF)
set(ACORN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ACORN" CACHE PATH "Path to the ACORN checkout.")
set(ACORN_BUILD_DIR "${ACORN_ROOT}/build" CACHE PATH "Path to the ACORN build directory.")

add_library(vectormaton ${VECTORMATON_CORE_SOURCES})
target_include_directories(vectormaton PUBLIC ${VECTORMATON_INCLUDE_DIRS})
target_link_libraries(vectormaton PUBLIC OpenMP::OpenMP_CXX)

# Create executable
add_executable(sa_test source/tests/test_sa.cpp)
add_executable(queue_test source/mpmc_queue.h source/tests/test_queue.cpp)
add_executable(hnsw_test source/headers.h source/tests/test_hnsw.cpp)
add_executable(vectormaton_test source/tests/test_vectormaton.cpp)
add_executable(main_exp source/headers.h source/experiments/main_exp.cpp ${VECTORMATON_BASELINE_SOURCES})
add_executable(postfiltering_case_study source/headers.h source/experiments/postfiltering_case_study.cpp source/baselines/exact.h source/baselines/exact.cpp source/baselines/post_filtering.h source/baselines/post_filtering.cpp)

foreach(target sa_test queue_test hnsw_test vectormaton_test main_exp postfiltering_case_study)
    target_include_directories(${target} PRIVATE ${VECTORMATON_INCLUDE_DIRS})
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
endforeach()

target_link_libraries(sa_test PRIVATE vectormaton)
target_link_libraries(vectormaton_test PRIVATE vectormaton)
target_link_libraries(main_exp PRIVATE vectormaton)


if(ENABLE_ACORN)
    find_library(ACORN_FAISS_LIBRARY
        NAMES faiss
        PATHS "${ACORN_BUILD_DIR}/faiss"
        NO_DEFAULT_PATH
        REQUIRED)
    find_package(ZLIB REQUIRED)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)

    add_executable(acorn_exp source/experiments/acorn_exp.cpp)
    target_include_directories(acorn_exp PRIVATE ${VECTORMATON_INCLUDE_DIRS} "${ACORN_ROOT}")
    target_link_libraries(acorn_exp
        vectormaton
        "${ACORN_FAISS_LIBRARY}"
        OpenMP::OpenMP_CXX
        ZLIB::ZLIB
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES})
    set_target_properties(acorn_exp PROPERTIES
        BUILD_RPATH "${ACORN_BUILD_DIR}/faiss")
endif()
