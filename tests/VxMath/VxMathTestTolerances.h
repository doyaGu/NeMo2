/**
 * @file VxMathTestTolerances.h
 * @brief Centralized tolerance constants and comparison utilities for VxMath tests.
 *
 * This header provides standardized tolerance values for different types of
 * floating-point operations, along with comparison utilities that handle
 * both absolute and scale-relative tolerances.
 */

#ifndef VXMATH_TEST_TOLERANCES_H
#define VXMATH_TEST_TOLERANCES_H

#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <limits>

namespace VxMathTest {

//=============================================================================
// Tolerance Constants
//=============================================================================

/// Strictest tolerance for exact operations (constructors, accessors, identity)
constexpr float STRICT_TOL = 1e-6f;

/// Tolerance for SIMD vs scalar comparison (expected minor rounding differences)
constexpr float SIMD_TOL = 5e-7f;

/// Standard tolerance for normal floating-point operations
constexpr float STANDARD_TOL = 1e-5f;

/// Tolerance for binary compatibility with ground-truth VxMath.dll (circa 2002)
constexpr float BINARY_TOL = 1e-4f;

/// Tolerance for operations with error accumulation (matrix chains, interpolations)
constexpr float ACCUMULATION_TOL = 1e-3f;

/// Loose tolerance for decomposition/reconstruction and complex pipelines
constexpr float LOOSE_TOL = 5e-3f;

//=============================================================================
// Tolerance Selection
//=============================================================================

/// Classification of operations for automatic tolerance selection
enum class ToleranceClass {
    Exact,           ///< Constructor, copy, assignment, identity
    SIMD,            ///< SIMD vs scalar comparison
    Standard,        ///< Normal arithmetic operations
    Binary,          ///< Binary compatibility testing
    Accumulated,     ///< Chain operations with error accumulation
    Loose            ///< Decomposition, reconstruction, complex pipelines
};

/// Get the appropriate tolerance value for a given operation class
constexpr float GetTolerance(ToleranceClass tc) {
    switch (tc) {
        case ToleranceClass::Exact:       return STRICT_TOL;
        case ToleranceClass::SIMD:        return SIMD_TOL;
        case ToleranceClass::Standard:    return STANDARD_TOL;
        case ToleranceClass::Binary:      return BINARY_TOL;
        case ToleranceClass::Accumulated: return ACCUMULATION_TOL;
        case ToleranceClass::Loose:       return LOOSE_TOL;
        default:                          return STANDARD_TOL;
    }
}

//=============================================================================
// Comparison Utilities
//=============================================================================

/**
 * @brief Absolute tolerance comparison.
 * @param a First value
 * @param b Second value
 * @param tol Absolute tolerance threshold
 * @return true if |a - b| < tol
 */
inline bool AbsoluteNear(float a, float b, float tol = STANDARD_TOL) {
    return std::fabs(a - b) < tol;
}

/**
 * @brief Scale-relative tolerance comparison.
 *
 * This is superior to absolute tolerance for values with large magnitude
 * differences. The tolerance scales with the magnitude of the values.
 *
 * From SIMDDispatchConsistencyTest.cpp - proven robust for SIMD validation.
 *
 * @param a First value
 * @param b Second value
 * @param tol Relative tolerance factor
 * @return true if the values are within the scale-relative tolerance
 */
inline bool ScaleRelativeNear(float a, float b, float tol = STANDARD_TOL) {
    const float diff = std::fabs(a - b);
    const float scale = 1.0f + std::max(std::fabs(a), std::fabs(b));
    return diff <= tol * scale;
}

/**
 * @brief ULP (Units in Last Place) comparison.
 *
 * Most accurate method for floating-point equality. Compares the number
 * of representable floating-point values between a and b.
 *
 * @param a First value
 * @param b Second value
 * @param maxUlps Maximum allowed ULP difference
 * @return true if the values are within maxUlps of each other
 */
inline bool UlpNear(float a, float b, int maxUlps = 4) {
    // Handle exact equality (also handles infinities)
    if (a == b) return true;

    // Handle NaN cases
    if (std::isnan(a) || std::isnan(b)) return false;

    // Different signs means not equal (except for +0 and -0 handled above)
    if ((a < 0) != (b < 0)) return false;

    // Reinterpret as integers for ULP comparison
    int32_t aInt, bInt;
    std::memcpy(&aInt, &a, sizeof(int32_t));
    std::memcpy(&bInt, &b, sizeof(int32_t));

    return std::abs(aInt - bInt) <= maxUlps;
}

/**
 * @brief Combined comparison with fallback strategies.
 *
 * First tries absolute comparison, then scale-relative if that fails.
 * Useful for general-purpose tolerance checking.
 *
 * @param a First value
 * @param b Second value
 * @param absTol Absolute tolerance
 * @param relTol Relative tolerance factor
 * @return true if values match either tolerance criterion
 */
inline bool Near(float a, float b, float absTol = STANDARD_TOL, float relTol = STANDARD_TOL) {
    // Fast path: absolute comparison
    if (std::fabs(a - b) < absTol) return true;

    // Fallback: scale-relative comparison
    return ScaleRelativeNear(a, b, relTol);
}

/**
 * @brief Check if a float is approximately zero.
 * @param a Value to check
 * @param tol Tolerance for zero
 * @return true if |a| < tol
 */
inline bool IsNearZero(float a, float tol = STANDARD_TOL) {
    return std::fabs(a) < tol;
}

/**
 * @brief Check if a float is approximately one.
 * @param a Value to check
 * @param tol Tolerance for one
 * @return true if |a - 1| < tol
 */
inline bool IsNearOne(float a, float tol = STANDARD_TOL) {
    return std::fabs(a - 1.0f) < tol;
}

/**
 * @brief Check if a value is finite (not NaN or Infinity).
 * @param a Value to check
 * @return true if finite
 */
inline bool IsFinite(float a) {
    return std::isfinite(a);
}

/**
 * @brief Check if a value is valid for math operations (finite and reasonable).
 * @param a Value to check
 * @param maxMagnitude Maximum allowed absolute value
 * @return true if value is finite and within bounds
 */
inline bool IsValidFloat(float a, float maxMagnitude = 1e20f) {
    return std::isfinite(a) && std::fabs(a) < maxMagnitude;
}

} // namespace VxMathTest

#endif // VXMATH_TEST_TOLERANCES_H
