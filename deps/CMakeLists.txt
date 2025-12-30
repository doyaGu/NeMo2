# =============================================================================
# External Dependencies
# =============================================================================

# -----------------------------------------------------------------------------
# miniz - Compression library
# -----------------------------------------------------------------------------
add_subdirectory(miniz EXCLUDE_FROM_ALL)
set_target_properties(miniz PROPERTIES FOLDER "Dependencies")

# -----------------------------------------------------------------------------
# STB - Header-only image library
# -----------------------------------------------------------------------------
set(STB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/stb")
if(NOT EXISTS "${STB_INCLUDE_DIR}")
    message(WARNING "STB headers not found at ${STB_INCLUDE_DIR}")
else()
    add_library(stb INTERFACE)
    add_library(NeMo2::STB ALIAS stb)
    target_include_directories(stb INTERFACE
        $<BUILD_INTERFACE:${STB_INCLUDE_DIR}>
    )
endif()

# -----------------------------------------------------------------------------
# SIMDE - SIMD everywhere (portable SIMD)
# -----------------------------------------------------------------------------
set(SIMDE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/simde")
if(NOT EXISTS "${SIMDE_INCLUDE_DIR}")
    message(WARNING "SIMDE headers not found at ${SIMDE_INCLUDE_DIR}")
else()
    add_library(simde INTERFACE)
    add_library(NeMo2::SIMDE ALIAS simde)
    target_include_directories(simde INTERFACE
        $<BUILD_INTERFACE:${SIMDE_INCLUDE_DIR}>
    )
endif()