cmake_minimum_required(VERSION 3.9)
project(MatrixMul C)
#set(CMAKE_BUILD_TYPE "Release")
set(ENVIRONMENT "CC")
add_definitions(-DWITH_AVX2)
add_definitions(-mavx2)
set(CMAKE_C_STANDARD 11)
add_compile_options(-mavx2 -fopenmp)
add_compile_options(-O3)

FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    link_libraries(-fopenmp)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
    link_libraries(-lblas)
    add_definitions(-DUNIX)
    add_definitions(-DWITH_BLAS)
    add_executable(main main.c Matmul.c BlasMatMul.c)
    MESSAGE(STATUS "current platform: Linux ")
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Windows")
    add_definitions(-DWINDOWS)
    add_executable(main main.c Matmul.c)
    MESSAGE(STATUS "current platform: Windows")
ENDIF()

add_executable(test BlasMatMul.c)




