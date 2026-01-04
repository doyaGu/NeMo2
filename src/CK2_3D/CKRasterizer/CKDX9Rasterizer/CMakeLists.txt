option(NEMO_RASTERIZER_USE_D3D9EX "Use DirectX 9Ex rasterizer path" OFF)
option(NEMO_RASTERIZER_ENABLE_TRIPLE_BUFFER "Enable triple buffering in rasterizer" OFF)

if (WIN32)
    set(CKDX9_RASTERIZER_SOURCES
            CKDX9Rasterizer.cpp
            CKDX9RasterizerContext.cpp
            CKDX9RasterizerDriver.cpp
    )

    nemo_add_library(
            NAME CKDX9Rasterizer
            CXX_STANDARD 17
            SOURCES ${CKDX9_RASTERIZER_SOURCES}
            PRIVATE_DEPS
            NeMo2::CKRasterizerLib
            d3d9
            dxguid
    )

    if (NEMO_RASTERIZER_USE_D3D9EX)
        target_compile_definitions(NeMo2::CKDX9Rasterizer PRIVATE USE_D3D9EX)
    endif ()

    if (NEMO_RASTERIZER_ENABLE_TRIPLE_BUFFER)
        target_compile_definitions(NeMo2::CKDX9Rasterizer PRIVATE ENABLE_TRIPLE_BUFFER)
    endif ()
endif ()
