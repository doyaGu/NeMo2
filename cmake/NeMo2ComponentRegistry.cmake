# Virtools runtime module declarations.
#
# Component repositories keep their own standalone CMake declarations. This
# registry only describes how the engine aggregator assembles and verifies them.

include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/NeMo2Modules.cmake")

set(NEMO_CORE_RUNTIME_TARGETS ${_nemo_core_modules})

# Keep the supported catalogue separate from the configured engine. Consumers
# decide which selected modules their application requires.
function(nemo_resolve_runtime_group group)
    set(selected)
    set(outputs)
    set(NEMO_SUPPORTED_${group}_TARGETS "${NEMO_${group}_TARGETS}" PARENT_SCOPE)
    foreach (name IN LISTS NEMO_${group}_TARGETS)
        if (TARGET ${name})
            set(target ${name})
        elseif (TARGET ${name}Static)
            set(target ${name}Static)
        else ()
            continue()
        endif ()
        get_target_property(output ${target} OUTPUT_NAME)
        if (NOT output)
            set(output ${name})
        endif ()
        list(APPEND selected ${name})
        list(APPEND outputs ${output})
    endforeach ()
    set(NEMO_${group}_TARGETS "${selected}" PARENT_SCOPE)
    set(NEMO_${group}_OUTPUTS "${outputs}" PARENT_SCOPE)
endfunction()

set(NEMO_RENDER_ENGINE_RUNTIME_TARGETS ${_nemo_render_modules})
get_property(_nemo_supported_render_targets GLOBAL PROPERTY CKRE_SUPPORTED_RUNTIME_TARGETS)
get_property(NEMO_RENDER_ENGINE_DISABLED_OUTPUTS GLOBAL PROPERTY CKRE_DISABLED_RUNTIME_OUTPUTS)
get_property(NEMO_RENDER_ENGINE_RUNTIME_CONFIGS GLOBAL PROPERTY CKRE_RUNTIME_CONFIGS)

set(NEMO_MANAGER_RUNTIME_TARGETS ${_nemo_manager_modules})
set(NEMO_PLUGIN_RUNTIME_TARGETS ${_nemo_plugin_modules})
set(NEMO_BUILDING_BLOCK_RUNTIME_TARGETS ${_nemo_bb_modules})
foreach (group IN ITEMS CORE_RUNTIME RENDER_ENGINE_RUNTIME MANAGER_RUNTIME
        PLUGIN_RUNTIME BUILDING_BLOCK_RUNTIME)
    nemo_resolve_runtime_group(${group})
endforeach ()
if (_nemo_supported_render_targets)
    set(NEMO_SUPPORTED_RENDER_ENGINE_RUNTIME_TARGETS ${_nemo_supported_render_targets})
endif ()

# Compatibility names describe selected outputs, not application requirements.
set(NEMO_BUILDING_BLOCK_REQUIRED_OUTPUTS ${NEMO_BUILDING_BLOCK_RUNTIME_OUTPUTS})
set(NEMO_BUILDING_BLOCK_OPTIONAL_OUTPUTS "")
if (NOT NEMO_RENDER_ENGINE_RUNTIME_TARGETS)
    set(NEMO_RENDER_ENGINE_RUNTIME_CONFIGS "")
endif ()

set(NEMO_MODULE_RUNTIME_TARGETS
        ${NEMO_RENDER_ENGINE_RUNTIME_TARGETS}
        ${NEMO_MANAGER_RUNTIME_TARGETS}
        ${NEMO_PLUGIN_RUNTIME_TARGETS}
        ${NEMO_BUILDING_BLOCK_RUNTIME_TARGETS}
)
set(NEMO_SUPPORTED_MODULE_RUNTIME_TARGETS
        ${NEMO_SUPPORTED_RENDER_ENGINE_RUNTIME_TARGETS}
        ${NEMO_SUPPORTED_MANAGER_RUNTIME_TARGETS}
        ${NEMO_SUPPORTED_PLUGIN_RUNTIME_TARGETS}
        ${NEMO_SUPPORTED_BUILDING_BLOCK_RUNTIME_TARGETS})

set(NEMO_RUNTIME_TARGETS
        ${NEMO_CORE_RUNTIME_TARGETS}
        ${NEMO_MODULE_RUNTIME_TARGETS}
)
if (NEMO_MODULE_SELECTION_EXPLICIT)
    foreach (_module IN LISTS NEMO_SELECTED_MODULES)
        if (NOT _module IN_LIST NEMO_RUNTIME_TARGETS)
            message(FATAL_ERROR "Selected NeMo2 module '${_module}' produced no target; check its shared/static build options")
        endif ()
    endforeach ()
endif ()

set(_nemo_component_manifest "${CMAKE_CURRENT_BINARY_DIR}/NeMo2RuntimeComponents.cmake")
configure_file("${CMAKE_CURRENT_LIST_DIR}/NeMo2RuntimeComponents.cmake.in"
        "${_nemo_component_manifest}" @ONLY)
set_property(GLOBAL PROPERTY NEMO_RUNTIME_COMPONENT_MANIFEST "${_nemo_component_manifest}")
