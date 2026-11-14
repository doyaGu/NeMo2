# Virtools runtime module declarations.
#
# Component repositories keep their own standalone CMake declarations. This
# registry only describes how the engine aggregator assembles and verifies them.

include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/NeMo2Modules.cmake")

set(NEMO_CORE_RUNTIME_TARGETS ${_nemo_core_modules})

function(nemo_library_prefix module result)
    if (module STREQUAL "VxMath")
        set(prefix VXMATH)
    elseif (module STREQUAL "CK2")
        set(prefix CK2)
    elseif (module IN_LIST _nemo_render_modules)
        set(prefix CKRE)
    elseif (module STREQUAL "SdlInputManager")
        set(prefix SDLINPUT)
    elseif (module STREQUAL "SdlSoundManager")
        set(prefix SDLSOUND)
    elseif (module STREQUAL "ParameterOperations")
        set(prefix CKPARAMOP)
    elseif (module IN_LIST _nemo_plugin_modules)
        set(prefix CKPLUGINS)
    else ()
        set(prefix CKBB)
    endif ()
    set(${result} "${prefix}" PARENT_SCOPE)
endfunction()

function(nemo_runtime_libraries module result)
    set(libraries)
    if (NOT module IN_LIST NEMO_TEST_ONLY_MODULES)
        nemo_library_prefix("${module}" prefix)
        foreach (form SHARED STATIC)
            set(target "${module}")
            if (form STREQUAL "STATIC")
                string(APPEND target Static)
            endif ()
            if (TARGET ${target})
                # Without aggregator options, describe the supplied targets.
                if (NOT DEFINED NEMO_RUNTIME_${prefix}_BUILD_${form} OR
                        NEMO_RUNTIME_${prefix}_BUILD_${form})
                    get_target_property(alias ${target} ALIASED_TARGET)
                    if (alias)
                        set(target "${alias}")
                    endif ()
                    list(APPEND libraries "${target}")
                endif ()
            endif ()
        endforeach ()
    endif ()
    list(REMOVE_DUPLICATES libraries)
    set(${result} "${libraries}" PARENT_SCOPE)
endfunction()

# Keep the supported catalogue separate from the configured engine. Consumers
# decide which selected modules their application requires.
function(nemo_resolve_runtime_group group)
    set(selected)
    set(outputs)
    set(NEMO_SUPPORTED_${group}_TARGETS "${NEMO_${group}_TARGETS}" PARENT_SCOPE)
    foreach (name IN LISTS NEMO_${group}_TARGETS)
        nemo_runtime_libraries("${name}" libraries)
        if (NOT libraries)
            continue()
        endif ()
        list(GET libraries 0 target)
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
foreach (_module IN LISTS _nemo_core_modules)
    if (NOT _module IN_LIST NEMO_RUNTIME_TARGETS)
        message(FATAL_ERROR "Required NeMo2 core module '${_module}' has no runtime library")
    endif ()
endforeach ()

# Validate legacy AUTO combinations as well as explicitly resolved selections.
foreach (_module IN LISTS NEMO_RUNTIME_TARGETS)
    nemo_module_dependencies("${_module}" _dependencies)
    foreach (_dependency IN LISTS _dependencies)
        if (NOT _dependency IN_LIST NEMO_RUNTIME_TARGETS)
            message(FATAL_ERROR "NeMo2 module '${_module}' requires runtime module '${_dependency}'; enable it or use an explicit NEMO_MODULES selection")
        endif ()
    endforeach ()
endforeach ()

set(NEMO_RUNTIME_LIBRARY_TARGETS "")
set(_nemo_runtime_details "")
foreach (_module IN LISTS NEMO_RUNTIME_TARGETS)
    nemo_runtime_libraries("${_module}" _libraries)
    nemo_module_dependencies("${_module}" _dependencies)
    list(APPEND NEMO_RUNTIME_LIBRARY_TARGETS ${_libraries})
    string(APPEND _nemo_runtime_details
            "set(NEMO_MODULE_${_module}_LIBRARY_TARGETS \"${_libraries}\")\n"
            "set(NEMO_MODULE_${_module}_REQUIRES \"${_dependencies}\")\n")
endforeach ()
set(NEMO_TEST_ONLY_TARGETS "")
foreach (_module IN LISTS _nemo_all_modules)
    foreach (_target "${_module}" "${_module}Static")
        if (TARGET ${_target})
            get_target_property(_alias ${_target} ALIASED_TARGET)
            if (_alias)
                set(_target "${_alias}")
            endif ()
            if (NOT _target IN_LIST NEMO_RUNTIME_LIBRARY_TARGETS)
                list(APPEND NEMO_TEST_ONLY_TARGETS "${_target}")
            endif ()
        endif ()
    endforeach ()
endforeach ()
list(REMOVE_DUPLICATES NEMO_TEST_ONLY_TARGETS)
foreach (_role RUNTIME TEST_ONLY)
    set(_targets "${NEMO_${_role}_TARGETS}")
    if (_role STREQUAL "RUNTIME")
        set(_targets "${NEMO_RUNTIME_LIBRARY_TARGETS}")
    endif ()
    foreach (_target IN LISTS _targets)
        get_target_property(_type ${_target} TYPE)
        string(APPEND _nemo_runtime_details
                "set(NEMO_LIBRARY_${_target}_TYPE \"${_type}\")\n"
                "set(NEMO_LIBRARY_${_target}_ROLE \"${_role}\")\n")
    endforeach ()
endforeach ()

set(_nemo_component_manifest "${CMAKE_CURRENT_BINARY_DIR}/NeMo2RuntimeComponents.cmake")
configure_file("${CMAKE_CURRENT_LIST_DIR}/NeMo2RuntimeComponents.cmake.in"
        "${_nemo_component_manifest}" @ONLY)
set_property(GLOBAL PROPERTY NEMO_RUNTIME_COMPONENT_MANIFEST "${_nemo_component_manifest}")
