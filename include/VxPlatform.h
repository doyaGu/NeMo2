#ifndef VXPLATFORM_H
#define VXPLATFORM_H

// Platform / compiler / architecture feature macros.
//
// Preferred source: provided by the build system (CMake add_compile_definitions).
// Fallback: detect from standard predefined macros so the headers remain usable
// outside the main build.

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

#ifndef VX_PLATFORM_POSIX
#   if !VX_PLATFORM_WINDOWS
#       define VX_PLATFORM_POSIX 1
#   else
#       define VX_PLATFORM_POSIX 0
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

#endif // VXPLATFORM_H
