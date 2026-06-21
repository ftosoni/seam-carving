cmake_minimum_required(VERSION 3.12)
project(seam_carving CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP)

add_executable(seam_carving
    src/main.cpp
    src/seam_carving.cpp
    src/seam_carving.h
    src/visualise.cpp
    src/visualise.h
)

target_include_directories(seam_carving PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
)

if(OpenMP_FOUND)
    target_link_libraries(seam_carving PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(seam_carving PRIVATE USE_OPENMP)
    message(STATUS "OpenMP found and enabled")
else()
    message(WARNING "OpenMP not found. Compiling without parallel support.")
endif()

# Build Unit Tests
option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    enable_testing()
    add_executable(seam_carving_tests
        tests/tests.cpp
        src/seam_carving.cpp
        src/seam_carving.h
        src/visualise.cpp
        src/visualise.h
    )
    target_include_directories(seam_carving_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty
    )
    if(OpenMP_FOUND)
        target_link_libraries(seam_carving_tests PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(seam_carving_tests PRIVATE USE_OPENMP)
    endif()
    add_test(NAME SeamCarvingTests COMMAND seam_carving_tests)
endif()
