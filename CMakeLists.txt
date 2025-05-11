cmake_minimum_required(VERSION 3.18)
project(JaccardSimilarity LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD   17)
set(CMAKE_CUDA_STANDARD  17)
set(CMAKE_CUDA_FLAGS     "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_ARCHITECTURES 75)

set(SRC_DIR   ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(TEST_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/tests)

include_directories(${SRC_DIR})
include_directories(${TEST_DIR})

# ─── 1. Kernel library: glob *.cu then exclude main.cu ────────────────────────
file(GLOB ALL_SRC_CU
    "${SRC_DIR}/*.cu"
)
# remove main.cu and reduce.cu so it's not compiled into the object library
list(REMOVE_ITEM ALL_SRC_CU "${SRC_DIR}/main.cu")
list(REMOVE_ITEM ALL_SRC_CU "${SRC_DIR}/reduce.cu")

add_library(jaccard_kernels OBJECT ${ALL_SRC_CU})
set_target_properties(jaccard_kernels
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)

# ─── 2. Production app (now main.cu only appears once) ───────────────────────
# add_executable(jaccard_app
    # $<TARGET_OBJECTS:jaccard_kernels>
    # ${SRC_DIR}/main.cu
# )
# set_target_properties(jaccard_app
    # PROPERTIES CUDA_SEPARABLE_COMPILATION ON
# )

add_executable(reduce_sum_app
    $<TARGET_OBJECTS:jaccard_kernels>
    ${SRC_DIR}/reduce.cu
)
set_target_properties(reduce_sum_app
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)

# ─── 3. Unit tests ───────────────────────────────────────────────────────────
option(BUILD_TESTS "Build CUDA unit tests" ON)
if (BUILD_TESTS)

enable_testing()

file(GLOB TEST_SOURCES
"${TEST_DIR}/test_*.cu"
)
    # remove test_main.cu so it's not compiled into the object library
    list(REMOVE_ITEM ALL_SRC_CU "${TEST_DIR}/test_main.cu")

    add_executable(jaccard_tests
    $<TARGET_OBJECTS:jaccard_kernels>
    ${TEST_SOURCES}
    )
    set_target_properties(jaccard_tests
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
    )
    target_compile_definitions(jaccard_tests PRIVATE BUILD_TESTS)

    add_test(NAME JaccardUnitTests COMMAND jaccard_tests)
endif()
