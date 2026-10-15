cmake_minimum_required(VERSION 3.16)
project(SeqPositionProbs CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_compile_options(-Wall -fno-math-errno -fno-trapping-math)
# Target architecture, used to pick the portable baseline ABI below.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_dummest_is_aarch64 TRUE)
else()
    set(_dummest_is_aarch64 FALSE)
endif()

# Portable container builds: -DDUMMEST_PORTABLE=ON pins a baseline ISA and a
# matching Kokkos ABI without injecting a native flag, so one binary runs on a
# whole family of hosts. x86-64 -> -march=x86-64-v3 + AVX2 (Haswell, 2013+);
# aarch64 -> -march=armv8-a + NEON (the 64-bit ARM baseline). The default
# (OFF) builds for the native host. DUMMEST_MARCH overrides -march= (empty
# disables).
option(DUMMEST_PORTABLE "Build portable baseline-ISA binaries for containers" OFF)
if(DUMMEST_PORTABLE)
    if(_dummest_is_aarch64)
        set(_dummest_default_march "armv8-a")
    else()
        set(_dummest_default_march "x86-64-v3")
    endif()
else()
    set(_dummest_default_march "native")
endif()
set(DUMMEST_MARCH "${_dummest_default_march}" CACHE STRING "Value for -march= (empty disables)")
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

# DUMMEST_PORTABLE pins Kokkos to a baseline ABI (AVX2/Haswell on x86-64,
# ARMv8/NEON on aarch64); otherwise use the host's native ABI and let Kokkos
# autodetect the SIMD backend (e.g. ARM NEON on aarch64).
if(DUMMEST_PORTABLE)
    set(Kokkos_ARCH_NATIVE OFF CACHE BOOL "" FORCE)
    if(_dummest_is_aarch64)
        set(Kokkos_ARCH_ARMV80 ON CACHE BOOL "" FORCE)
    else()
        set(Kokkos_ARCH_HSW ON CACHE BOOL "" FORCE)
    endif()
elseif(NOT DEFINED Kokkos_ARCH_NATIVE)
    set(Kokkos_ARCH_NATIVE ON CACHE BOOL "")
endif()
add_subdirectory(kokkos-5.1.1)

# Isolated builds (e.g. profiling runs that own bin/dummestl_core) can pass
# -DDUMMEST_SKIP_BIN_COPY=ON so building never touches bin/.
option(DUMMEST_SKIP_BIN_COPY "Skip copying built binaries to bin/" OFF)

# Debug "poison" mode: fill uninitialized DP scratch with NaN sentinels so
# stale/previous-batch/OOB reads abort via assert (see dummest-poison.hh).
# Debug-only: never shipped. A poison build that finishes with output
# identical to a normal build proves every scratch read was preceded by a write.
# Poison binaries must use a scratch DUMMEST_CACHE_DIR (calibration results
# from poison runs are still valid scores, but keep caches separate anyway).
option(DUMMEST_POISON "Fill uninitialized DP scratch with NaN poison (debug only)" OFF)

set(DUMMEST_SOURCES
    dummest.cc dummest-util.hh dummest-score.hh dummest-poison.hh dummest-batch.hh
    dummest-scalerow.hh dummest-vectraits.hh dummest-padded.hh dummest-core.hh
    dummest-sequence.hh dummest-output.hh dummest-dp.hh dummest-traceback.hh
    dummest-dp-backward.hh dummest-dp-forward.hh dummest-search.hh
    dummest-calibration.hh dummest-profile.hh tantan-wrapper.hh version.hh
    can_i_haz_simd.hh)

add_executable(dummest_core ${DUMMEST_SOURCES})
target_compile_definitions(dummest_core PRIVATE DOUBLE FILTER_PASS)
target_include_directories(dummest_core PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(dummest_core Kokkos::kokkos)
if(NOT DUMMEST_SKIP_BIN_COPY)
add_custom_command(TARGET dummest_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:dummest_core> "${CMAKE_SOURCE_DIR}/bin/dummest_core")
endif()

add_executable(dummestl_core ${DUMMEST_SOURCES})
target_compile_definitions(dummestl_core PRIVATE FILTER_PASS)
target_include_directories(dummestl_core PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(dummestl_core Kokkos::kokkos)
if(DUMMEST_POISON)
    target_compile_definitions(dummest_core PRIVATE DUMMEST_POISON)
    target_compile_definitions(dummestl_core PRIVATE DUMMEST_POISON)
endif()
if(NOT DUMMEST_SKIP_BIN_COPY)
add_custom_command(TARGET dummestl_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:dummestl_core> "${CMAKE_SOURCE_DIR}/bin/dummestl_core")
endif()

add_executable(dummest-build dummest-build.cc dummest-util.hh version.hh priors/gap-priors.hh priors/wheeler4.hh priors/blocks9.hh)
if(NOT DUMMEST_SKIP_BIN_COPY)
add_custom_command(TARGET dummest-build POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:dummest-build> "${CMAKE_SOURCE_DIR}/bin/dummest-build")
endif()

# Unit tests (compile-only; Kokkos is header-only here). Built in the same tree
# so their tuning flags and SIMD width match the engine.
add_executable(test_batch_pairs tests/test_batch_pairs.cc)
target_include_directories(test_batch_pairs PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(test_batch_pairs PRIVATE Kokkos::kokkos)
if(DUMMEST_POISON)
    target_compile_definitions(test_batch_pairs PRIVATE DUMMEST_POISON)
endif()
