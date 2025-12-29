#ifndef VXMATHCOMPILER_H
#define VXMATHCOMPILER_H

#include "VxPlatform.h"

/**
 * @file VxMathCompiler.h
 * @brief Compiler-specific macros for VxMath.
 *
 * This header provides compiler-specific macros for:
 * - DLL export/import declarations
 * - Calling conventions
 * - Alignment and attributes
 * - C++ standard detection
 */

// ---------------------------------------------------------------------------
// Compiler / platform / arch detection
//
// Preferred source: CMake add_compile_definitions(...) provides VX_* macros.
// Fallback: detect from standard predefined macros to keep headers usable
// outside the CMake build.
// ---------------------------------------------------------------------------

#ifndef VX_COMPILER_MSVC
#   if defined(_MSC_VER) && !defined(__clang__)
#       define VX_COMPILER_MSVC 1
#   else
#       define VX_COMPILER_MSVC 0
#   endif
#endif

#ifndef VX_COMPILER_CLANG
#   if defined(__clang__)
#       define VX_COMPILER_CLANG 1
#   else
#       define VX_COMPILER_CLANG 0
#   endif
#endif

#ifndef VX_COMPILER_GCC
#   if defined(__GNUC__) && !defined(__clang__)
#       define VX_COMPILER_GCC 1
#   else
#       define VX_COMPILER_GCC 0
#   endif
#endif

#ifndef VX_PLATFORM_WINDOWS
#   if defined(_WIN32)
#       define VX_PLATFORM_WINDOWS 1
#   else
#       define VX_PLATFORM_WINDOWS 0
#   endif
#endif

#ifndef VX_ARCH_X86
#   if defined(_M_IX86) || defined(__i386__)
#       define VX_ARCH_X86 1
#   else
#       define VX_ARCH_X86 0
#   endif
#endif

#ifndef VX_ARCH_X64
#   if defined(_M_X64) || defined(__x86_64__) || defined(__amd64__)
#       define VX_ARCH_X64 1
#   else
#       define VX_ARCH_X64 0
#   endif
#endif

// Legacy helper macros (keep for existing code that might rely on them)
#if defined(_MSC_VER)
#   define VX_MSVC _MSC_VER
#elif defined(__GNUC__)
#   define VX_GCC __GNUC__
#endif

#if defined(_MSC_VER) // Microsoft Visual C++
#   if _MSC_VER < 1200 // Visual Studio 6.0
#       error "Unsupported compiler."
#   elif _MSC_VER >= 1400 // .Net 2005 and higher
#       pragma warning(disable : 4996)
#   endif
#endif

#ifndef VX_EXPORT
#   ifdef VX_LIB
#       define VX_EXPORT
#   else
#       if VX_PLATFORM_WINDOWS
#           ifdef VX_API
#               define VX_EXPORT __declspec(dllexport)
#           else
#               define VX_EXPORT __declspec(dllimport)
#           endif
#       else
            // GCC/Clang visibility attribute for shared libraries
#           define VX_EXPORT __attribute__((visibility("default")))
#       endif
#   endif // VX_LIB
#endif // !VX_EXPORT

#define CK_PRIVATE_VERSION_VIRTOOLS

// EXPORT DEFINES FOR LIB / DLL VERSIONS
#ifndef CK_LIB
#   ifdef CK_PRIVATE_VERSION_VIRTOOLS
#       if VX_PLATFORM_WINDOWS
#           define DLL_EXPORT __declspec(dllexport)
#       else
#           define DLL_EXPORT __attribute__((visibility("default")))
#       endif
#   else
#       define DLL_EXPORT
#   endif
#else
#   define DLL_EXPORT
#endif

#ifndef CK_LIB
#   if VX_PLATFORM_WINDOWS
#       define PLUGIN_EXPORT extern "C" __declspec(dllexport)
#   else
#       define PLUGIN_EXPORT extern "C" __attribute__((visibility("default")))
#   endif
#else
#   define PLUGIN_EXPORT
#endif // CK_LIB

#ifdef __cplusplus
#   define BEGIN_CDECLS extern "C" {
#   define END_CDECLS }
#else
#   define BEGIN_CDECLS
#   define END_CDECLS
#endif

#ifndef VX_NOEXCEPT
#   if (__cplusplus >= 201103L) || (defined(_MSC_VER) && _MSC_VER >= 1900)
#       define VX_NOEXCEPT noexcept
#   else
#       define VX_NOEXCEPT throw()
#   endif
#endif

#ifndef VX_DEPRECATED
#   if defined(__cplusplus)
#       if defined(__has_cpp_attribute)
#           if __has_cpp_attribute(deprecated)
#               define VX_DEPRECATED [[deprecated]]
#           endif
#       endif
#   endif
#   if !defined(VX_DEPRECATED)
#       if VX_COMPILER_MSVC
#           define VX_DEPRECATED __declspec(deprecated)
#       elif VX_COMPILER_GCC || VX_COMPILER_CLANG
#           define VX_DEPRECATED __attribute__((deprecated))
#       else
#           define VX_DEPRECATED
#       endif
#   endif
#endif

#ifndef VX_NAKED
#   if defined(_MSC_VER)
#       define VX_NAKED __declspec(naked)
#   elif defined(__GNUC__)
#       define VX_NAKED __attribute__((naked))
#   else
#       define VX_NAKED
#   endif
#endif

// Calling conventions
#ifndef VX_CDECL
#   if VX_PLATFORM_WINDOWS
#       define VX_CDECL __cdecl
#   else
#       define VX_CDECL
#   endif
#endif

#ifndef VX_FASTCALL
#   if VX_PLATFORM_WINDOWS
#       define VX_FASTCALL __fastcall
#   else
#       define VX_FASTCALL
#   endif
#endif

#ifndef VX_STDCALL
#   if VX_PLATFORM_WINDOWS
#       define VX_STDCALL __stdcall
#   else
#       define VX_STDCALL
#   endif
#endif

#ifndef VX_THISCALL
#   if VX_PLATFORM_WINDOWS
#       define VX_THISCALL __thiscall
#   else
#       define VX_THISCALL
#   endif
#endif

#ifndef VX_ALIGN
#   if defined(_MSC_VER)
#       define VX_ALIGN(x) __declspec(align(x))
#   elif defined(__GNUC__)
#       define VX_ALIGN(x) __attribute__((aligned(x)))
#   endif
#endif

#ifndef VX_SECTION
#   if defined(_MSC_VER)
#       define VX_SECTION(x) __declspec(code_seg(x))
#   elif defined(__GNUC__)
#       define VX_SECTION(x) __attribute__((section(x)))
#   endif
#endif

#ifndef VX_SELECTANY
#   if defined(_MSC_VER)
#       define VX_SELECTANY __declspec(selectany)
#   elif defined(__GNUC__)
#       define VX_SELECTANY
#   endif
#endif

#ifndef VX_HAS_CXX11
#   if (__cplusplus >= 201103L) || (defined(_MSVC_LANG) && (_MSVC_LANG >= 201103L)) || (defined(_MSC_VER) && (_MSC_VER >= 1900))
#       define VX_HAS_CXX11 1
#   else
#       define VX_HAS_CXX11 0
#   endif
#endif

#endif // VXMATHCOMPILER_H
