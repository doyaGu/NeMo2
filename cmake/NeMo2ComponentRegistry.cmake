# Virtools runtime module declarations.
#
# Component repositories keep their own standalone CMake declarations. This
# registry only describes how the engine aggregator assembles and verifies them.

include_guard(GLOBAL)

set(NEMO_CORE_RUNTIME_TARGETS
        VxMath CK2
)

get_property(NEMO_RENDER_ENGINE_RUNTIME_TARGETS GLOBAL PROPERTY CKRE_RUNTIME_TARGETS)
get_property(NEMO_RENDER_ENGINE_DISABLED_OUTPUTS GLOBAL PROPERTY CKRE_DISABLED_RUNTIME_OUTPUTS)
get_property(NEMO_RENDER_ENGINE_RUNTIME_CONFIGS GLOBAL PROPERTY CKRE_RUNTIME_CONFIGS)
set(NEMO_RENDER_ENGINE_RUNTIME_OUTPUTS ${NEMO_RENDER_ENGINE_RUNTIME_TARGETS})

set(NEMO_MANAGER_RUNTIME_TARGETS
        SdlInputManager SdlSoundManager ParameterOperations
)
set(NEMO_MANAGER_RUNTIME_OUTPUTS
        SdlInputManager SdlSoundManager ParameterOperations
)

set(NEMO_PLUGIN_RUNTIME_TARGETS
        AVIReader ImageReader WavReader VirtoolsLoader
)
set(NEMO_PLUGIN_RUNTIME_OUTPUTS
        AVIReader ImageReader WavReader VirtoolsLoader
)

set(NEMO_BUILDING_BLOCK_RUNTIME_TARGETS
        3DTrans BuildingBlocksAddons1 Cameras Characters Collision Controllers
        Grids Interface Lights Logics Materials MeshModifiers MidiManager Narratives
        Sounds Visuals WorldEnvironment physics_RT TT_DatabaseManager_RT
        TT_Gravity_RT TT_InterfaceManager_RT TT_ParticleSystems_RT TT_Toolbox_RT
)
set(NEMO_BUILDING_BLOCK_REQUIRED_OUTPUTS
        3DTransfo BuildingBlocksAddons1 Cameras Characters Collisions Controllers
        Grids Interface Lights Logics Materials MeshModifiers Narratives Sounds
        Visuals WorldEnvironment physics_RT TT_DatabaseManager_RT TT_Gravity_RT
        TT_InterfaceManager_RT TT_ParticleSystems_RT TT_Toolbox_RT
)
set(NEMO_BUILDING_BLOCK_OPTIONAL_OUTPUTS
        MidiManager
)

set(NEMO_MODULE_RUNTIME_TARGETS
        ${NEMO_RENDER_ENGINE_RUNTIME_TARGETS}
        ${NEMO_MANAGER_RUNTIME_TARGETS}
        ${NEMO_PLUGIN_RUNTIME_TARGETS}
        ${NEMO_BUILDING_BLOCK_RUNTIME_TARGETS}
)

set(NEMO_RUNTIME_TARGETS
        ${NEMO_CORE_RUNTIME_TARGETS}
        ${NEMO_MODULE_RUNTIME_TARGETS}
)

set(_nemo_component_manifest "${CMAKE_CURRENT_BINARY_DIR}/NeMo2RuntimeComponents.cmake")
configure_file("${CMAKE_CURRENT_LIST_DIR}/NeMo2RuntimeComponents.cmake.in"
        "${_nemo_component_manifest}" @ONLY)
set_property(GLOBAL PROPERTY NEMO_RUNTIME_COMPONENT_MANIFEST "${_nemo_component_manifest}")
