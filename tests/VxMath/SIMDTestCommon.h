/**
 * @file SIMDTestCommon.h
 * @brief Common utilities, fixtures, and helpers for VxSIMD test suite.
 *
 * This header provides:
 * - Shared test fixtures for SIMD tests
 * - Reference (scalar) implementations for comparison
 * - Common test data generators
 * - Tolerance and comparison utilities specific to SIMD
 */

#ifndef SIMD_TEST_COMMON_H
#define SIMD_TEST_COMMON_H

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>
#include <array>
#include <algorithm>

#include "VxSIMD.h"
#include "VxMath.h"
#include "VxMathDefines.h"
#include "VxVector.h"
#include "VxMatrix.h"
#include "VxQuaternion.h"
#include "VxRay.h"
#include "VxPlane.h"
#include "VxRect.h"
#include "VxFrustum.h"
#include "Vx2dVector.h"

#include "VxMathTestHelpers.h"
#include "VxMathTestTolerances.h"
#include "VxMathTestGenerators.h"

namespace SIMDTest {

//=============================================================================
// SIMD-Specific Tolerances
//=============================================================================

/// Tolerance for SIMD vs scalar comparisons (accounts for rsqrt approximation, FMA differences)
constexpr float SIMD_SCALAR_TOL = 1e-5f;

/// Tighter tolerance for operations that should be nearly exact (add, subtract, etc.)
constexpr float SIMD_EXACT_TOL = 1e-6f;

/// Looser tolerance for operations with accumulated error (chains, normalizations)
constexpr float SIMD_ACCUMULATED_TOL = 1e-4f;

/// Tolerance for angle-related operations (radians)
constexpr float SIMD_ANGLE_TOL = 1e-5f;

//=============================================================================
// Comparison Utilities
//=============================================================================

inline bool NearEqual(float a, float b, float tol = SIMD_SCALAR_TOL) {
    const float diff = std::fabs(a - b);
    const float scale = 1.0f + std::max(std::fabs(a), std::fabs(b));
    return diff <= tol * scale;
}

inline bool VectorNear(const VxVector& a, const VxVector& b, float tol = SIMD_SCALAR_TOL) {
    return NearEqual(a.x, b.x, tol) && NearEqual(a.y, b.y, tol) && NearEqual(a.z, b.z, tol);
}

inline bool Vector4Near(const VxVector4& a, const VxVector4& b, float tol = SIMD_SCALAR_TOL) {
    return NearEqual(a.x, b.x, tol) && NearEqual(a.y, b.y, tol) &&
           NearEqual(a.z, b.z, tol) && NearEqual(a.w, b.w, tol);
}

inline bool QuaternionNear(const VxQuaternion& a, const VxQuaternion& b, float tol = SIMD_SCALAR_TOL) {
    // Quaternions q and -q represent the same rotation
    const bool same = NearEqual(a.x, b.x, tol) && NearEqual(a.y, b.y, tol) &&
                     NearEqual(a.z, b.z, tol) && NearEqual(a.w, b.w, tol);
    if (same) return true;

    return NearEqual(a.x, -b.x, tol) && NearEqual(a.y, -b.y, tol) &&
           NearEqual(a.z, -b.z, tol) && NearEqual(a.w, -b.w, tol);
}

inline bool MatrixNear(const VxMatrix& a, const VxMatrix& b, float tol = SIMD_SCALAR_TOL) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (!NearEqual(a[i][j], b[i][j], tol)) {
                return false;
            }
        }
    }
    return true;
}

inline bool RectNear(const VxRect& a, const VxRect& b, float tol = SIMD_SCALAR_TOL) {
    return NearEqual(a.left, b.left, tol) && NearEqual(a.top, b.top, tol) &&
           NearEqual(a.right, b.right, tol) && NearEqual(a.bottom, b.bottom, tol);
}

inline bool PlaneNear(const VxPlane& a, const VxPlane& b, float tol = SIMD_SCALAR_TOL) {
    return NearEqual(a.m_Normal.x, b.m_Normal.x, tol) && NearEqual(a.m_Normal.y, b.m_Normal.y, tol) &&
           NearEqual(a.m_Normal.z, b.m_Normal.z, tol) && NearEqual(a.m_D, b.m_D, tol);
}

//=============================================================================
// EXPECT Macros for SIMD Tests
//=============================================================================

#define EXPECT_SIMD_NEAR(a, b, tol)  \
    EXPECT_TRUE(SIMDTest::NearEqual(a, b, tol)) << \
        "Expected: " << (a) << " vs " << (b) << " (tol=" << (tol) << ")"

#define EXPECT_SIMD_VEC3_NEAR(a, b, tol) \
    EXPECT_TRUE(SIMDTest::VectorNear(a, b, tol)) << \
        "Vector mismatch: (" << (a).x << ", " << (a).y << ", " << (a).z << ") vs (" \
        << (b).x << ", " << (b).y << ", " << (b).z << ")"

#define EXPECT_SIMD_VEC4_NEAR(a, b, tol) \
    EXPECT_TRUE(SIMDTest::Vector4Near(a, b, tol)) << \
        "Vector4 mismatch: (" << (a).x << ", " << (a).y << ", " << (a).z << ", " << (a).w << ") vs (" \
        << (b).x << ", " << (b).y << ", " << (b).z << ", " << (b).w << ")"

#define EXPECT_SIMD_QUAT_NEAR(a, b, tol) \
    EXPECT_TRUE(SIMDTest::QuaternionNear(a, b, tol)) << \
        "Quaternion mismatch: (" << (a).x << ", " << (a).y << ", " << (a).z << ", " << (a).w << ") vs (" \
        << (b).x << ", " << (b).y << ", " << (b).z << ", " << (b).w << ")"

#define EXPECT_SIMD_MAT_NEAR(a, b, tol) \
    EXPECT_TRUE(SIMDTest::MatrixNear(a, b, tol)) << "Matrix mismatch"

//=============================================================================
// Scalar Reference Implementations
//=============================================================================

namespace Scalar {

inline void AddVector(VxVector& result, const VxVector& a, const VxVector& b) {
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    result.z = a.z + b.z;
}

inline void SubtractVector(VxVector& result, const VxVector& a, const VxVector& b) {
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;
}

inline void ScaleVector(VxVector& result, const VxVector& v, float s) {
    result.x = v.x * s;
    result.y = v.y * s;
    result.z = v.z * s;
}

inline float DotProduct3(const VxVector& a, const VxVector& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline void CrossProduct(VxVector& result, const VxVector& a, const VxVector& b) {
    result.x = a.y * b.z - a.z * b.y;
    result.y = a.z * b.x - a.x * b.z;
    result.z = a.x * b.y - a.y * b.x;
}

inline float VectorLength(const VxVector& v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline float VectorLengthSquared(const VxVector& v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

inline float VectorDistance(const VxVector& a, const VxVector& b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

inline void VectorLerp(VxVector& result, const VxVector& a, const VxVector& b, float t) {
    result.x = a.x + (b.x - a.x) * t;
    result.y = a.y + (b.y - a.y) * t;
    result.z = a.z + (b.z - a.z) * t;
}

inline void VectorReflect(VxVector& result, const VxVector& incident, const VxVector& normal) {
    const float dot = 2.0f * DotProduct3(incident, normal);
    result.x = incident.x - dot * normal.x;
    result.y = incident.y - dot * normal.y;
    result.z = incident.z - dot * normal.z;
}

inline void VectorMinimize(VxVector& result, const VxVector& a, const VxVector& b) {
    result.x = std::min(a.x, b.x);
    result.y = std::min(a.y, b.y);
    result.z = std::min(a.z, b.z);
}

inline void VectorMaximize(VxVector& result, const VxVector& a, const VxVector& b) {
    result.x = std::max(a.x, b.x);
    result.y = std::max(a.y, b.y);
    result.z = std::max(a.z, b.z);
}

inline void NormalizeVector(VxVector& v) {
    const float lenSq = v.x * v.x + v.y * v.y + v.z * v.z;
    if (lenSq > EPSILON) {
        const float invLen = 1.0f / std::sqrt(lenSq);
        v.x *= invLen;
        v.y *= invLen;
        v.z *= invLen;
    }
}

// Vector4 operations
inline void AddVector4(VxVector4& result, const VxVector4& a, const VxVector4& b) {
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    result.z = a.z + b.z;
    result.w = a.w + b.w;
}

inline void SubtractVector4(VxVector4& result, const VxVector4& a, const VxVector4& b) {
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    result.z = a.z - b.z;
    result.w = a.w - b.w;
}

inline void ScaleVector4(VxVector4& result, const VxVector4& v, float s) {
    result.x = v.x * s;
    result.y = v.y * s;
    result.z = v.z * s;
    result.w = v.w * s;
}

inline float DotProduct4(const VxVector4& a, const VxVector4& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline void LerpVector4(VxVector4& result, const VxVector4& a, const VxVector4& b, float t) {
    result.x = a.x + (b.x - a.x) * t;
    result.y = a.y + (b.y - a.y) * t;
    result.z = a.z + (b.z - a.z) * t;
    result.w = a.w + (b.w - a.w) * t;
}

// Pixel operations
inline int ConvertPixelBatch32(const XULONG* srcPixels, XULONG* dstPixels, int count, const VxPixelSimdConfig& config) {
    if (!config.enabled) return 0;

    const int simdCount = count & ~3;
    if (simdCount <= 0) return 0;

    for (int i = 0; i < simdCount; ++i) {
        XULONG dst = config.alphaFill ? config.alphaFillComponent : 0u;
        const XULONG src = srcPixels[i];

        for (int c = 0; c < 4; ++c) {
            if (!config.channelCopy[c]) continue;
            XULONG channel = src & config.srcMasks[c];
            if (config.srcShiftRight[c]) {
                channel >>= static_cast<XULONG>(config.srcShiftRight[c]);
            }
            if (config.dstShiftLeft[c]) {
                channel <<= static_cast<XULONG>(config.dstShiftLeft[c]);
            }
            channel &= config.dstMasks[c];
            dst |= channel;
        }
        dstPixels[i] = dst;
    }
    return simdCount;
}

inline int ApplyAlphaBatch32(XULONG* pixels, int count, XBYTE alphaValue, XULONG alphaMask, XULONG alphaShift) {
    const int simdCount = count & ~3;
    if (simdCount <= 0) return 0;

    const XULONG alphaComponent = (static_cast<XULONG>(alphaValue) << alphaShift) & alphaMask;
    const XULONG colorMask = ~alphaMask;

    for (int i = 0; i < simdCount; ++i) {
        pixels[i] = (pixels[i] & colorMask) | alphaComponent;
    }
    return simdCount;
}

inline int ApplyVariableAlphaBatch32(XULONG* pixels, const XBYTE* alphaValues, int count, XULONG alphaMask, XULONG alphaShift) {
    const int simdCount = count & ~3;
    if (simdCount <= 0) return 0;

    const XULONG colorMask = ~alphaMask;
    for (int i = 0; i < simdCount; ++i) {
        const XULONG a = (static_cast<XULONG>(alphaValues[i]) << alphaShift) & alphaMask;
        pixels[i] = (pixels[i] & colorMask) | a;
    }
    return simdCount;
}

} // namespace Scalar

//=============================================================================
// Test Fixtures
//=============================================================================

/**
 * @brief Base fixture for SIMD tests with random generator and dispatch access.
 */
class SIMDTestBase : public ::testing::Test {
protected:
    void SetUp() override {
        m_dispatch = VxGetSIMDDispatch();
        ASSERT_NE(m_dispatch, nullptr) << "SIMD dispatch table is null";
        m_rng.SetSeed(0xDEADBEEF);
    }

    const VxSIMDDispatch* m_dispatch = nullptr;
    VxMathTest::RandomGenerator m_rng;

    // Helper to get random vectors
    VxVector RandomVector(float min = -100.0f, float max = 100.0f) {
        return m_rng.Vector(min, max);
    }

    VxVector RandomUnitVector() {
        return m_rng.UnitVector();
    }

    VxVector RandomSmallVector() {
        return m_rng.SmallVector(1e-6f);
    }

    VxVector4 RandomVector4(float min = -100.0f, float max = 100.0f) {
        return m_rng.Vector4(min, max);
    }

    VxMatrix RandomTRSMatrix() {
        return m_rng.TRSMatrix();
    }

    VxMatrix RandomRotationMatrix() {
        return m_rng.RotationMatrix();
    }

    VxQuaternion RandomQuaternion() {
        return m_rng.ArbitraryQuaternion();
    }

    VxQuaternion RandomUnitQuaternion() {
        return m_rng.UnitQuaternion();
    }

    float RandomFloat(float min = -100.0f, float max = 100.0f) {
        return m_rng.Float(min, max);
    }

    float RandomAngle() {
        return m_rng.Angle();
    }

    float RandomInterpolationFactor() {
        return m_rng.InterpolationFactor();
    }
};

/**
 * @brief Fixture for testing aligned memory operations.
 */
class SIMDAlignedMemoryTest : public ::testing::Test {
protected:
    void TearDown() override {
        for (void* ptr : m_allocatedPtrs) {
            VxAlignedFree(ptr);
        }
        m_allocatedPtrs.clear();
    }

    void* AllocAndTrack(size_t size, size_t alignment) {
        void* ptr = VxAlignedMalloc(size, alignment);
        if (ptr) m_allocatedPtrs.push_back(ptr);
        return ptr;
    }

    std::vector<void*> m_allocatedPtrs;
};

//=============================================================================
// Test Data Generators
//=============================================================================

/**
 * @brief Generates test vectors including edge cases.
 */
inline std::vector<VxVector> GenerateTestVectors() {
    std::vector<VxVector> vectors;

    // Standard test vectors
    vectors.push_back(VxVector(1.0f, 0.0f, 0.0f));   // Unit X
    vectors.push_back(VxVector(0.0f, 1.0f, 0.0f));   // Unit Y
    vectors.push_back(VxVector(0.0f, 0.0f, 1.0f));   // Unit Z
    vectors.push_back(VxVector(1.0f, 1.0f, 1.0f));   // Diagonal
    vectors.push_back(VxVector(-1.0f, -1.0f, -1.0f)); // Negative diagonal

    // Edge cases
    vectors.push_back(VxVector(0.0f, 0.0f, 0.0f));   // Zero vector
    vectors.push_back(VxVector(1e-6f, 1e-6f, 1e-6f)); // Very small
    vectors.push_back(VxVector(1e6f, 1e6f, 1e6f));   // Large values
    vectors.push_back(VxVector(-1e6f, 1e6f, -1e6f)); // Mixed large

    // Random values
    VxMathTest::RandomGenerator rng(42);
    for (int i = 0; i < 10; ++i) {
        vectors.push_back(rng.Vector(-100.0f, 100.0f));
    }

    return vectors;
}

/**
 * @brief Generates test matrices including edge cases.
 */
inline std::vector<VxMatrix> GenerateTestMatrices() {
    std::vector<VxMatrix> matrices;
    VxMathTest::RandomGenerator rng(42);

    // Identity
    matrices.push_back(rng.IdentityMatrix());

    // Rotations
    matrices.push_back(rng.RotationMatrix());
    matrices.push_back(rng.RotationMatrix());

    // Translation only
    matrices.push_back(rng.TranslationMatrix());

    // Scale only
    matrices.push_back(rng.ScaleMatrix());
    matrices.push_back(rng.NonUniformScaleMatrix());

    // Combined TRS
    for (int i = 0; i < 5; ++i) {
        matrices.push_back(rng.TRSMatrix());
    }

    return matrices;
}

/**
 * @brief Generates test quaternions including edge cases.
 */
inline std::vector<VxQuaternion> GenerateTestQuaternions() {
    std::vector<VxQuaternion> quats;
    VxMathTest::RandomGenerator rng(42);

    // Identity
    quats.push_back(VxQuaternion(0.0f, 0.0f, 0.0f, 1.0f));

    // 90 degree rotations around axes
    quats.push_back(VxQuaternion(0.707107f, 0.0f, 0.0f, 0.707107f));  // X
    quats.push_back(VxQuaternion(0.0f, 0.707107f, 0.0f, 0.707107f));  // Y
    quats.push_back(VxQuaternion(0.0f, 0.0f, 0.707107f, 0.707107f));  // Z

    // Random unit quaternions
    for (int i = 0; i < 10; ++i) {
        quats.push_back(rng.UnitQuaternion());
    }

    return quats;
}

} // namespace SIMDTest

#endif // SIMD_TEST_COMMON_H
