cmake_minimum_required(VERSION 3.16)
project(SeqPositionProbs CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_compile_options(-Wall -march=native -fno-math-errno -fno-trapping-math)
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

set(Kokkos_ARCH_NATIVE ON CACHE BOOL "" FORCE)
add_subdirectory(kokkos-5.1.1)

# Isolated builds (e.g. profiling runs that own bin/dummerl) can pass
# -DDUMMER_SKIP_BIN_COPY=ON so building never touches bin/.
option(DUMMER_SKIP_BIN_COPY "Skip copying built binaries to bin/" OFF)

add_executable(dummer dummer.cc dummer-util.hh dummer-score.hh dummer-batch.hh dummer-scalerow.hh dummer-vectraits.hh dummer-padded.hh dummer-core.hh dummer-sequence.hh dummer-output.hh dummer-dp.hh dummer-traceback.hh dummer-dp-backward.hh dummer-dp-forward.hh dummer-search.hh dummer-calibration.hh dummer-profile.hh tantan-wrapper.hh version.hh can_i_haz_simd.hh)
target_compile_definitions(dummer PRIVATE DOUBLE FILTER_PASS)
target_include_directories(dummer PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(dummer Kokkos::kokkos)
if(NOT DUMMER_SKIP_BIN_COPY)
add_custom_command(TARGET dummer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:dummer> "${CMAKE_SOURCE_DIR}/bin/dummer")
endif()

add_executable(dummerl dummer.cc dummer-util.hh dummer-score.hh dummer-batch.hh dummer-scalerow.hh dummer-vectraits.hh dummer-padded.hh dummer-core.hh dummer-sequence.hh dummer-output.hh dummer-dp.hh dummer-traceback.hh dummer-dp-backward.hh dummer-dp-forward.hh dummer-search.hh dummer-calibration.hh dummer-profile.hh tantan-wrapper.hh version.hh can_i_haz_simd.hh)
target_compile_definitions(dummerl PRIVATE FILTER_PASS)
target_include_directories(dummerl PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(dummerl Kokkos::kokkos)
if(NOT DUMMER_SKIP_BIN_COPY)
add_custom_command(TARGET dummerl POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:dummerl> "${CMAKE_SOURCE_DIR}/bin/dummerl")
endif()

add_executable(dummer-build dummer-build.cc dummer-util.hh version.hh priors/gap-priors.hh priors/wheeler4.hh priors/blocks9.hh)
if(NOT DUMMER_SKIP_BIN_COPY)
add_custom_command(TARGET dummer-build POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:dummer-build> "${CMAKE_SOURCE_DIR}/bin/dummer-build")
endif()
