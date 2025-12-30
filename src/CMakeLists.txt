# =============================================================================
# NeMo2 Source Libraries
# =============================================================================

add_subdirectory(VxMath)
add_subdirectory(CK2)
add_subdirectory(CK2_3D)

# =============================================================================
# Installation and Export (centralized)
# =============================================================================

# Export all library targets together
install(EXPORT NeMo2Targets
    FILE NeMo2Targets.cmake
    NAMESPACE NeMo2::
    DESTINATION "${NEMO_INSTALL_CMAKEDIR}"
)