# =============================================================================
# CK2_3D Rasterizer pieces
# =============================================================================

set(CKRASTERIZER_LIB_SOURCES
        CKRasterizer.cpp
        CKRasterizerDriver.cpp
        CKRasterizerContext.cpp
)

nemo_add_library(
        NAME CKRasterizerLib
        TYPE STATIC
        SOURCES ${CKRASTERIZER_LIB_SOURCES}
        PUBLIC_DEPS
        NeMo2::VxMath
        NeMo2::CompileOptions
)