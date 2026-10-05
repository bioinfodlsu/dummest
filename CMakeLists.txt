cmake_minimum_required(VERSION 3.16)
project(SeqPositionProbs CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_compile_options(-Wall -fno-math-errno -fno-trapping-math)
# Portable container builds: -DDUMMEST_PORTABLE=ON selects -march=x86-64-v3 and
# pins Kokkos to its AVX2 (Haswell) ABI without injecting -march=native, matching
# can_i_haz_simd.hh. Requires Haswell/2013+ (x86-64-v3). The default (OFF) builds
# for the native host. DUMMEST_MARCH overrides -march= directly (empty disables).
option(DUMMEST_PORTABLE "Build portable x86-64-v3 (AVX2) binaries for containers" OFF)
if(DUMMEST_PORTABLE)
    set(_dummest_default_march "x86-64-v3")
else()
    set(_dummest_default_march "native")
endif()
set(DUMMEST_MARCH "${_dummest_default_march}" CACHE STRING "Value for -march= flag (empty disables)")
if(DUMMEST_MARCH)
    add_compile_options(-march=${DUMMEST_MARCH})
endif()
add_compile_options(
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
        $<$<CONFIG:Debug>:-fno-omit-frame-pointer>
)
add_link_options(
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
)

if(NOT CMAKE_BUILD_TYPE MATCHES "Debug")
    string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")

    # MSVC specific flag handling (uses forward slash)
    if(MSVC)
        string(REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
        string(REPLACE "/DNDEBUG" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
    endif()
endif()

find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --dirty
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
else()
    set(GIT_VERSION "unknown")
endif()

file(WRITE "${CMAKE_BINARY_DIR}/version.hh.tmp" "\"${GIT_VERSION}\"\n")
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_BINARY_DIR}/version.hh.tmp" "${CMAKE_SOURCE_DIR}/version.hh")

# DUMMEST_PORTABLE pins Kokkos to the AVX2 (Haswell) ABI; otherwise use the host.
if(DUMMEST_PORTABLE)
    set(Kokkos_ARCH_NATIVE OFF CACHE BOOL "" FORCE)
    set(Kokkos_ARCH_HSW ON CACHE BOOL "" FORCE)
elseif(NOT DEFINED Kokkos_ARCH_NATIVE)
    set(Kokkos_ARCH_NATIVE ON CACHE BOOL "")
endif()
add_subdirectory(kokkos-5.1.1)

# Isolated builds (e.g. profiling runs that own bin/dummestl) can pass
# -DDUMMEST_SKIP_BIN_COPY=ON so building never touches bin/.
option(DUMMEST_SKIP_BIN_COPY "Skip copying built binaries to bin/" OFF)

add_executable(dummest dummest.cc dummest-util.hh dummest-score.hh dummest-batch.hh dummest-scalerow.hh dummest-vectraits.hh dummest-padded.hh dummest-core.hh dummest-sequence.hh dummest-output.hh dummest-dp.hh dummest-traceback.hh dummest-dp-backward.hh dummest-dp-forward.hh dummest-search.hh dummest-calibration.hh dummest-profile.hh tantan-wrapper.hh version.hh can_i_haz_simd.hh)
target_compile_definitions(dummest PRIVATE DOUBLE FILTER_PASS)
target_include_directories(dummest PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(dummest Kokkos::kokkos)
if(NOT DUMMEST_SKIP_BIN_COPY)
add_custom_command(TARGET dummest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:dummest> "${CMAKE_SOURCE_DIR}/bin/dummest")
endif()

add_executable(dummestl dummest.cc dummest-util.hh dummest-score.hh dummest-batch.hh dummest-scalerow.hh dummest-vectraits.hh dummest-padded.hh dummest-core.hh dummest-sequence.hh dummest-output.hh dummest-dp.hh dummest-traceback.hh dummest-dp-backward.hh dummest-dp-forward.hh dummest-search.hh dummest-calibration.hh dummest-profile.hh tantan-wrapper.hh version.hh can_i_haz_simd.hh)
target_compile_definitions(dummestl PRIVATE FILTER_PASS)
target_include_directories(dummestl PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(dummestl Kokkos::kokkos)
if(NOT DUMMEST_SKIP_BIN_COPY)
add_custom_command(TARGET dummestl POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:dummestl> "${CMAKE_SOURCE_DIR}/bin/dummestl")
endif()

add_executable(dummest-build dummest-build.cc dummest-util.hh version.hh priors/gap-priors.hh priors/wheeler4.hh priors/blocks9.hh)
if(NOT DUMMEST_SKIP_BIN_COPY)
add_custom_command(TARGET dummest-build POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:dummest-build> "${CMAKE_SOURCE_DIR}/bin/dummest-build")
endif()
