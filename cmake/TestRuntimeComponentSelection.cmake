cmake_minimum_required(VERSION 3.25)
set(root "${TEST_BINARY_DIR}/component-selection")
file(MAKE_DIRECTORY "${root}/source")
file(WRITE "${root}/source/CMakeLists.txt" "
cmake_minimum_required(VERSION 3.25)
project(ComponentSelection NONE)
add_library(VxMathStatic INTERFACE)
add_library(CK2Static INTERFACE)
add_library(3DTrans INTERFACE)
set_property(TARGET 3DTrans PROPERTY OUTPUT_NAME 3DTransfo)
add_library(SdlInputManagerStatic INTERFACE)
add_library(ImageReader INTERFACE)
set_property(GLOBAL PROPERTY CKRE_RUNTIME_TARGETS CK2_3D)
set_property(GLOBAL PROPERTY CKRE_SUPPORTED_RUNTIME_TARGETS CK2_3D CKSdlGpuRasterizer)
set_property(GLOBAL PROPERTY CKRE_RUNTIME_CONFIGS CK2_3D.ini)
include(\"${TEST_SOURCE_DIR}/cmake/NeMo2ComponentRegistry.cmake\")
if (NOT NEMO_BUILDING_BLOCK_RUNTIME_TARGETS STREQUAL \"3DTrans\" OR
    NOT NEMO_BUILDING_BLOCK_RUNTIME_OUTPUTS STREQUAL \"3DTransfo\" OR
    NOT NEMO_MANAGER_RUNTIME_TARGETS STREQUAL \"SdlInputManager\" OR
    NOT NEMO_PLUGIN_RUNTIME_TARGETS STREQUAL \"ImageReader\" OR
    NEMO_RENDER_ENGINE_RUNTIME_TARGETS OR NEMO_RENDER_ENGINE_RUNTIME_CONFIGS)
    message(FATAL_ERROR \"Runtime manifest includes disabled modules or loses static targets/output names\")
endif()
if (NOT \"CKSdlGpuRasterizer\" IN_LIST NEMO_SUPPORTED_RENDER_ENGINE_RUNTIME_TARGETS OR
    NOT \"TT_InterfaceManager_RT\" IN_LIST NEMO_SUPPORTED_BUILDING_BLOCK_RUNTIME_TARGETS OR
    NOT \"AVIReader\" IN_LIST NEMO_SUPPORTED_PLUGIN_RUNTIME_TARGETS)
    message(FATAL_ERROR \"Supported catalogue lost disabled modules\")
endif()
get_property(manifest GLOBAL PROPERTY NEMO_RUNTIME_COMPONENT_MANIFEST)
include(\"\${manifest}\")
if (\"physics_RT\" IN_LIST NEMO_RUNTIME_TARGETS OR
    NOT \"SdlInputManager\" IN_LIST NEMO_MODULE_RUNTIME_TARGETS OR
    NOT NEMO_BUILDING_BLOCK_REQUIRED_OUTPUTS STREQUAL \"3DTransfo\")
    message(FATAL_ERROR \"Exported manifest differs from configured targets\")
endif()
")
execute_process(COMMAND "${CMAKE_COMMAND}" -S "${root}/source" -B "${root}/build"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if (NOT result EQUAL 0)
    message(FATAL_ERROR "Component selection failed:\n${output}${error}")
endif ()
