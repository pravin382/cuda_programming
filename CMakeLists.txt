cmake_minimum_required(VERSION 3.19.5)
project(CUDA_PRORAMMING LANGUAGES C CXX)

include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()

add_executable(main src/cmul_cublasxt.cu)

target_link_libraries(main cublas curand)
