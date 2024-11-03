cmake_minimum_required(VERSION 3.28.1)
project(bit_src)

find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    message("Found OpenMP")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "-fopenmp")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

endif()

IF (WIN32)
    set(CMAKE_EXE_LINKER_FLAGS "-static")
ENDIF()

set(CMAKE_CXX_FLAGS -O3)
set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(bit_src
        main.cpp
        position.hpp
        base.hpp
	    preGenMove.hpp
        genMove.hpp
        experiment.hpp
        evaluate.hpp
        search.hpp
        cache.hpp
        pipe.hpp
)

