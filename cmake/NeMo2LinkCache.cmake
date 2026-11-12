option(NEMO_ENABLE_LINK_CACHE "Reuse optimized code between LTO links" ON)
set(NEMO_LINK_CACHE_DIR "${CMAKE_BINARY_DIR}/.link-cache" CACHE PATH "Incremental LTO cache directory")

if (NOT NEMO_ENABLE_LINK_CACHE)
    return()
endif ()

include(CheckLinkerFlag)
if (APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    file(MAKE_DIRECTORY "${NEMO_LINK_CACHE_DIR}")
    set(_nemo_link_cache_flag "LINKER:-cache_path_lto,${NEMO_LINK_CACHE_DIR}")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 15)
    file(MAKE_DIRECTORY "${NEMO_LINK_CACHE_DIR}")
    set(_nemo_link_cache_flag "-flto-incremental=${NEMO_LINK_CACHE_DIR}")
elseif (MSVC)
    # Keep /GL optimization, and avoid restarting links when a static library
    # contains bitcode. The .iobj/.ipdb files are reusable with the build cache.
    set(_nemo_link_cache_flag "/LTCG:INCREMENTAL")
else ()
    return()
endif ()

check_linker_flag(CXX "${_nemo_link_cache_flag}" NEMO_LINK_CACHE_SUPPORTED)
if (NEMO_LINK_CACHE_SUPPORTED)
    add_link_options("$<$<CONFIG:Release>:${_nemo_link_cache_flag}>")
endif ()
