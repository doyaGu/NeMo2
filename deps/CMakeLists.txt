# miniz
add_subdirectory(miniz EXCLUDE_FROM_ALL)

# STB
set(STB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/stb" CACHE INTERNAL "STB include directory")
if(NOT EXISTS "${STB_INCLUDE_DIR}")
    message(WARNING "STB headers not found at ${STB_INCLUDE_DIR}. Please ensure STB is properly downloaded.")
endif()

# SIMDE
set(SIMDE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/simde" CACHE INTERNAL "SIMDE include directory")
if(NOT EXISTS "${SIMDE_INCLUDE_DIR}")
    message(WARNING "SIMDE headers not found at ${SIMDE_INCLUDE_DIR}. Please ensure SIMDE is properly downloaded.")
endif()