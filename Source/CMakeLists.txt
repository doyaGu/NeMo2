# Source - Component Aggregator
# Adds selected Virtools modules and configures output directories
# to match the Virtools runtime layout.

# --- Helpers ---

# Set Visual Studio folder for target
function(_set_vs_folder target folder_name)
    if (TARGET ${target})
        get_target_property(_aliased_target ${target} ALIASED_TARGET)
        if (_aliased_target)
            return()
        endif ()
        set_target_properties(${target} PROPERTIES FOLDER "${folder_name}")
    endif ()
endfunction()

# --- Components ---

# Core (VxMath, CK2) -> Bin/
message(STATUS "[NeMo2] Adding core components...")
add_subdirectory(VxMath)
add_subdirectory(CK2)
if (MSVC)
    _set_vs_folder(VxMath "Core")
    _set_vs_folder(VxMathStatic "Core")
    _set_vs_folder(CK2 "Core")
    _set_vs_folder(CK2Static "Core")
endif ()

# Render Engine -> RenderEngines/
if (_NEMO_WITH_RENDER_ENGINE)
    message(STATUS "[NeMo2] Adding render engine...")
    add_subdirectory(RenderEngine)
endif ()
if (MSVC)
    _set_vs_folder(CK2_3D "Core")
    _set_vs_folder(CK2_3DStatic "Core")
endif ()

# Managers -> Managers/
if (_NEMO_WITH_MANAGERS)
    message(STATUS "[NeMo2] Adding managers...")
    add_subdirectory(Managers)
endif ()
if (MSVC)
    foreach (_mgr IN ITEMS
            SdlInputManager SdlInputManagerStatic
            SdlSoundManager SdlSoundManagerStatic
            ParameterOperations ParameterOperationsStatic)
        _set_vs_folder(${_mgr} "Managers")
    endforeach ()
endif ()

# Plugins -> Plugins/
if (_NEMO_WITH_PLUGINS)
    message(STATUS "[NeMo2] Adding plugins...")
    add_subdirectory(Plugins)
endif ()
foreach (_plugin IN ITEMS AVIReader ImageReader WavReader VirtoolsLoader)
    if (MSVC)
        _set_vs_folder(${_plugin} "Plugins")
    endif ()
endforeach ()

# Building Blocks -> BuildingBlocks/
if (_NEMO_WITH_BUILDING_BLOCKS)
    message(STATUS "[NeMo2] Adding building blocks...")
    add_subdirectory(BuildingBlocks)
endif ()
if (MSVC)
    # Organize building block modules
    foreach (_bb_module IN LISTS _nemo_bb_modules)
        _set_vs_folder(${_bb_module} "BuildingBlocks")
        _set_vs_folder(${_bb_module}Static "BuildingBlocks")
    endforeach ()

    # Organize IVP physics library targets
    _set_vs_folder(ivp_utility "BuildingBlocks/IVP")
    _set_vs_folder(ivp_surface_manager "BuildingBlocks/IVP")
    _set_vs_folder(ivp_collision "BuildingBlocks/IVP")
    _set_vs_folder(ivp_intern "BuildingBlocks/IVP")
    _set_vs_folder(ivp_physics "BuildingBlocks/IVP")
    _set_vs_folder(ivp_controller "BuildingBlocks/IVP")
    _set_vs_folder(ivp_compact_builder "BuildingBlocks/IVP")
endif ()
