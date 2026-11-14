# Standalone installation uses the existing module Runtime/Development rules.
# Parent applications own their stage target and application runtime packaging.
get_property(_nemo_manifest GLOBAL PROPERTY NEMO_RUNTIME_COMPONENT_MANIFEST)
include("${_nemo_manifest}")

set(_nemo_sdl_runtime "")
if (TARGET SDL3::SDL3-shared)
    set(_nemo_sdl_runtime SDL3::SDL3-shared)
elseif (TARGET SDL3::SDL3)
    get_target_property(_nemo_sdl_runtime SDL3::SDL3 ALIASED_TARGET)
    if (NOT _nemo_sdl_runtime)
        set(_nemo_sdl_runtime SDL3::SDL3)
    endif ()
    get_target_property(_nemo_sdl_type ${_nemo_sdl_runtime} TYPE)
    if (NOT _nemo_sdl_type STREQUAL "SHARED_LIBRARY")
        set(_nemo_sdl_runtime "")
    endif ()
endif ()
if (_nemo_sdl_runtime)
    get_target_property(_nemo_sdl_framework ${_nemo_sdl_runtime} FRAMEWORK)
    if (_nemo_sdl_framework)
        message(FATAL_ERROR "The standalone stage requires an SDL3 shared-library package")
    endif ()
    install(IMPORTED_RUNTIME_ARTIFACTS ${_nemo_sdl_runtime}
            RUNTIME DESTINATION Bin COMPONENT Runtime
            LIBRARY DESTINATION Bin COMPONENT Runtime)
    if (APPLE)
        install(SCRIPT "${CMAKE_CURRENT_LIST_DIR}/RelocateMacOSRuntime.cmake"
                COMPONENT Runtime)
    endif ()
endif ()

add_custom_target(stage
        COMMAND "${CMAKE_COMMAND}" "-DNEMO_STAGE_BUILD_DIR=${CMAKE_BINARY_DIR}"
        "-DNEMO_STAGE_CONFIG=$<CONFIG>" -P "${CMAKE_CURRENT_LIST_DIR}/NeMo2StageInstall.cmake"
        COMMENT "Staging NeMo2 modules to ${CMAKE_BINARY_DIR}/stage"
        USES_TERMINAL VERBATIM)
foreach (_target IN LISTS NEMO_RUNTIME_TARGETS)
    foreach (_candidate IN ITEMS "${_target}" "${_target}Static")
        if (TARGET ${_candidate})
            add_dependencies(stage ${_candidate})
        endif ()
    endforeach ()
endforeach ()
if (TARGET ckre_scene_capture)
    add_dependencies(stage ckre_scene_capture)
endif ()
