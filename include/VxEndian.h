#ifndef VXENDIAN_H
#define VXENDIAN_H

/**
 * @file VxEndian.h
 * @brief Endianness detection and byte-swap utilities.
 *
 * This header provides:
 * - Endianness detection macros
 * - Byte-swap intrinsics for 16, 32, and 64-bit values
 * - Host to little-endian conversion functions
 * - Array conversion helpers for serialization
 *
 * All chunk data is stored in little-endian format for backward compatibility
 * with original Virtools files (Windows/x86).
 */

#include "VxMathDefines.h"

#if VX_COMPILER_MSVC
#   include <intrin.h>
#endif

// ============================================================================
// Endianness Detection
// ============================================================================

// CMake-based detection (preferred, set via add_compile_definitions)
#if !defined(VX_ENDIAN_BIG)
// Fallback: compiler/platform detection
#   if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__)
#       if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#           define VX_ENDIAN_BIG    1
#           define VX_ENDIAN_LITTLE 0
#       else
#           define VX_ENDIAN_BIG    0
#           define VX_ENDIAN_LITTLE 1
#       endif
#   elif VX_COMPILER_MSVC || VX_ARCH_X86 || VX_ARCH_X64 || \
         defined(__ARMEL__) || defined(__AARCH64EL__)
// Most common platforms are little-endian
#       define VX_ENDIAN_BIG    0
#       define VX_ENDIAN_LITTLE 1
#   elif defined(__ARMEB__) || defined(__AARCH64EB__) || defined(__MIPSEB__)
#       define VX_ENDIAN_BIG    1
#       define VX_ENDIAN_LITTLE 0
#   else
// Default to little-endian (Windows/x86 was the original target)
#       define VX_ENDIAN_BIG    0
#       define VX_ENDIAN_LITTLE 1
#   endif
#endif

// ============================================================================
// Byte-Swap Intrinsics
// ============================================================================

/**
 * @brief Swap bytes of a 16-bit value.
 * @param x The value to byte-swap.
 * @return The byte-swapped value.
 */
inline uint16_t VxByteSwap16(uint16_t x) {
#if VX_COMPILER_MSVC
    return _byteswap_ushort(x);
#elif VX_COMPILER_GCC || VX_COMPILER_CLANG
    return __builtin_bswap16(x);
#else
    return static_cast<uint16_t>((x >> 8) | (x << 8));
#endif
}

/**
 * @brief Swap bytes of a 32-bit value.
 * @param x The value to byte-swap.
 * @return The byte-swapped value.
 */
inline uint32_t VxByteSwap32(uint32_t x) {
#if VX_COMPILER_MSVC
    return _byteswap_ulong(x);
#elif VX_COMPILER_GCC || VX_COMPILER_CLANG
    return __builtin_bswap32(x);
#else
    return ((x >> 24) & 0x000000FF) |
           ((x >> 8)  & 0x0000FF00) |
           ((x << 8)  & 0x00FF0000) |
           ((x << 24) & 0xFF000000);
#endif
}

/**
 * @brief Swap bytes of a 64-bit value.
 * @param x The value to byte-swap.
 * @return The byte-swapped value.
 */
inline uint64_t VxByteSwap64(uint64_t x) {
#if VX_COMPILER_MSVC
    return _byteswap_uint64(x);
#elif VX_COMPILER_GCC || VX_COMPILER_CLANG
    return __builtin_bswap64(x);
#else
    return ((x >> 56) & 0x00000000000000FFULL) |
           ((x >> 40) & 0x000000000000FF00ULL) |
           ((x >> 24) & 0x0000000000FF0000ULL) |
           ((x >> 8)  & 0x00000000FF000000ULL) |
           ((x << 8)  & 0x000000FF00000000ULL) |
           ((x << 24) & 0x0000FF0000000000ULL) |
           ((x << 40) & 0x00FF000000000000ULL) |
           ((x << 56) & 0xFF00000000000000ULL);
#endif
}

// ============================================================================
// Little-Endian Conversion (for serialization)
// ============================================================================

/**
 * @brief Convert a 16-bit value from host byte order to little-endian.
 * @param x The value in host byte order.
 * @return The value in little-endian byte order.
 */
inline uint16_t VxHostToLE16(uint16_t x) {
#if VX_ENDIAN_LITTLE
    return x;
#else
    return VxByteSwap16(x);
#endif
}

/**
 * @brief Convert a 16-bit value from little-endian to host byte order.
 * @param x The value in little-endian byte order.
 * @return The value in host byte order.
 */
inline uint16_t VxLEToHost16(uint16_t x) {
#if VX_ENDIAN_LITTLE
    return x;
#else
    return VxByteSwap16(x);
#endif
}

/**
 * @brief Convert a 32-bit value from host byte order to little-endian.
 * @param x The value in host byte order.
 * @return The value in little-endian byte order.
 */
inline uint32_t VxHostToLE32(uint32_t x) {
#if VX_ENDIAN_LITTLE
    return x;
#else
    return VxByteSwap32(x);
#endif
}

/**
 * @brief Convert a 32-bit value from little-endian to host byte order.
 * @param x The value in little-endian byte order.
 * @return The value in host byte order.
 */
inline uint32_t VxLEToHost32(uint32_t x) {
#if VX_ENDIAN_LITTLE
    return x;
#else
    return VxByteSwap32(x);
#endif
}

/**
 * @brief Convert a 64-bit value from host byte order to little-endian.
 * @param x The value in host byte order.
 * @return The value in little-endian byte order.
 */
inline uint64_t VxHostToLE64(uint64_t x) {
#if VX_ENDIAN_LITTLE
    return x;
#else
    return VxByteSwap64(x);
#endif
}

/**
 * @brief Convert a 64-bit value from little-endian to host byte order.
 * @param x The value in little-endian byte order.
 * @return The value in host byte order.
 */
inline uint64_t VxLEToHost64(uint64_t x) {
#if VX_ENDIAN_LITTLE
    return x;
#else
    return VxByteSwap64(x);
#endif
}

// ============================================================================
// Array/Buffer Endian Conversion Helpers
// ============================================================================

/**
 * @brief Convert an array of 16-bit values from host to little-endian in-place.
 * @param data Pointer to the array of values.
 * @param count Number of elements in the array.
 */
inline void VxConvertArrayToLE16(void *data, size_t count) {
#if VX_ENDIAN_BIG
    uint16_t *arr = static_cast<uint16_t*>(data);
    for (size_t i = 0; i < count; ++i) {
        arr[i] = VxByteSwap16(arr[i]);
    }
#else
    (void)data;
    (void)count;
#endif
}

/**
 * @brief Convert an array of 16-bit values from little-endian to host in-place.
 * @param data Pointer to the array of values.
 * @param count Number of elements in the array.
 */
inline void VxConvertArrayFromLE16(void *data, size_t count) {
    VxConvertArrayToLE16(data, count); // Swap is symmetric
}

/**
 * @brief Convert an array of 32-bit values from host to little-endian in-place.
 * @param data Pointer to the array of values.
 * @param count Number of elements in the array.
 */
inline void VxConvertArrayToLE32(void *data, size_t count) {
#if VX_ENDIAN_BIG
    uint32_t *arr = static_cast<uint32_t*>(data);
    for (size_t i = 0; i < count; ++i) {
        arr[i] = VxByteSwap32(arr[i]);
    }
#else
    (void)data;
    (void)count;
#endif
}

/**
 * @brief Convert an array of 32-bit values from little-endian to host in-place.
 * @param data Pointer to the array of values.
 * @param count Number of elements in the array.
 */
inline void VxConvertArrayFromLE32(void *data, size_t count) {
    VxConvertArrayToLE32(data, count); // Swap is symmetric
}

/**
 * @brief Convert an array of 64-bit values from host to little-endian in-place.
 * @param data Pointer to the array of values.
 * @param count Number of elements in the array.
 */
inline void VxConvertArrayToLE64(void *data, size_t count) {
#if VX_ENDIAN_BIG
    uint64_t *arr = static_cast<uint64_t*>(data);
    for (size_t i = 0; i < count; ++i) {
        arr[i] = VxByteSwap64(arr[i]);
    }
#else
    (void)data;
    (void)count;
#endif
}

/**
 * @brief Convert an array of 64-bit values from little-endian to host in-place.
 * @param data Pointer to the array of values.
 * @param count Number of elements in the array.
 */
inline void VxConvertArrayFromLE64(void *data, size_t count) {
    VxConvertArrayToLE64(data, count); // Swap is symmetric
}

#endif // VXENDIAN_H
