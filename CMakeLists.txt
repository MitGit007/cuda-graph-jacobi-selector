cmake_minimum_required(VERSION 3.18)
project(JacobiSolver LANGUAGES CXX CUDA)

# ---- C++ / CUDA standards ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ---- GPU architecture: NVIDIA RTX 3050 Laptop GPU = Ampere sm_86 ----
# Installed CUDA Toolkit used for the primary experiments: 13.1
# Change this value if building for a different GPU architecture.
set(CMAKE_CUDA_ARCHITECTURES 86)

# ---- Compiler flags ----
# Use generator expressions so MSVC gets /O2 and GCC gets -O3.
set(CMAKE_CXX_FLAGS_RELEASE  "")   # cleared; set per-compiler below
set(CMAKE_CUDA_FLAGS         "${CMAKE_CUDA_FLAGS} -lineinfo")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -DNDEBUG")

# ---- Common kernel source (compiled ONCE, shared by all targets) ----
# jacobi_kernel.cu contains the __global__ definition; the .cuh has only
# the declaration.  Both targets link against the same compiled object via
# a static library to avoid nvlink multiple-definition errors.
add_library(jacobi_kernel STATIC src/role1/jacobi_kernel.cu)
target_include_directories(jacobi_kernel PUBLIC ${CMAKE_SOURCE_DIR}/src)
set_target_properties(jacobi_kernel PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(jacobi_kernel PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:$<IF:$<CXX_COMPILER_ID:MSVC>,/O2,-O3>>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# ---- Source files for the main executable ----
set(MAIN_SOURCES
    src/main.cu
    src/role1/cpu_jacobi.cpp
    src/role1/ordinary_cuda.cu
    src/role2/graph_cuda.cu
    src/role3/selector.cu
)

add_executable(jacobi ${MAIN_SOURCES})
target_include_directories(jacobi PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(jacobi PRIVATE jacobi_kernel)
target_compile_options(jacobi PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:$<IF:$<CXX_COMPILER_ID:MSVC>,/O2,-O3>>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
# ---- Correctness test executable ----
set(TEST_SOURCES
    tests/test_correctness.cu
    src/role1/cpu_jacobi.cpp
    src/role1/ordinary_cuda.cu
)

add_executable(jacobi_tests ${TEST_SOURCES})
target_include_directories(jacobi_tests PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(jacobi_tests PRIVATE jacobi_kernel)
target_compile_options(jacobi_tests PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:$<IF:$<CXX_COMPILER_ID:MSVC>,/O2,-O3>>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# ---- CUDA properties for all targets ----
foreach(tgt jacobi jacobi_tests)
    set_target_properties(${tgt} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endforeach()

# ---- Optional: install targets ----
install(TARGETS jacobi jacobi_tests DESTINATION bin)
