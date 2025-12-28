/**
 * @file VxMathTestHelpers.h
 * @brief Common test utilities and comparison macros for VxMath test suite.
 *
 * This header provides standardized tolerance values and comparison macros
 * to ensure consistent testing against the ground-truth VxMath.dll binary.
 *
 * New modular headers provide enhanced functionality:
 * - VxMathTestTolerances.h: Comprehensive tolerance constants and comparison utilities
 * - VxMathTestComparisons.h: Type-specific comparison functions with rich error messages
 * - VxMathTestGenerators.h: Random value generators for all VxMath types
 * - VxMathTestEdgeCases.h: Edge case and degenerate value generators
 * - VxMathTestPropertyTesting.h: Property-based testing infrastructure
 *
 * For new code, prefer using the VxMathTest:: namespace functions from the modular headers.
 */

#ifndef VXMATH_TEST_HELPERS_H
#define VXMATH_TEST_HELPERS_H

#include <gtest/gtest.h>
#include <cmath>
#include <string>
#include <sstream>
#include <random>

#include "VxMath.h"

//=============================================================================
// Include New Modular Headers
//=============================================================================

#include "VxMathTestTolerances.h"
#include "VxMathTestComparisons.h"
#include "VxMathTestGenerators.h"
#include "VxMathTestEdgeCases.h"
#include "VxMathTestPropertyTesting.h"

//=============================================================================
// Legacy Tolerance Constants (preserved for backward compatibility)
//=============================================================================

/**
 * @deprecated Use VxMathTest::BINARY_TOL or VxMathTest::ACCUMULATION_TOL instead.
 *
 * Standard tolerance for SIMD floating-point operations.
 * The ground-truth DLL (circa 2002) uses early SIMD implementations with
 * lower precision than modern scalar code. Typical error is ~1e-4.
 */
constexpr float VXMATH_TOLERANCE = 1e-3f;

/**
 * @deprecated Use VxMathTest::LOOSE_TOL instead.
 *
 * Loose tolerance for operations that accumulate error (e.g., matrix chains,
 * quaternion conversions, decompositions).
 */
constexpr float VXMATH_LOOSE_TOLERANCE = 5e-3f;

/**
 * @deprecated Use VxMathTest::STANDARD_TOL instead.
 *
 * Strict tolerance for simple operations that should be exact.
 */
constexpr float VXMATH_STRICT_TOLERANCE = 1e-5f;

//=============================================================================
// Vector Comparison Macros
//=============================================================================

#define EXPECT_VEC3_NEAR(v1, v2, tol) \
    do { \
        EXPECT_NEAR((v1).x, (v2).x, tol) << "VxVector.x mismatch"; \
        EXPECT_NEAR((v1).y, (v2).y, tol) << "VxVector.y mismatch"; \
        EXPECT_NEAR((v1).z, (v2).z, tol) << "VxVector.z mismatch"; \
    } while(0)

#define EXPECT_VEC3_EQ(v1, v2) EXPECT_VEC3_NEAR(v1, v2, VXMATH_TOLERANCE)

#define EXPECT_VEC4_NEAR(v1, v2, tol) \
    do { \
        EXPECT_NEAR((v1).x, (v2).x, tol) << "VxVector4.x mismatch"; \
        EXPECT_NEAR((v1).y, (v2).y, tol) << "VxVector4.y mismatch"; \
        EXPECT_NEAR((v1).z, (v2).z, tol) << "VxVector4.z mismatch"; \
        EXPECT_NEAR((v1).w, (v2).w, tol) << "VxVector4.w mismatch"; \
    } while(0)

#define EXPECT_VEC4_EQ(v1, v2) EXPECT_VEC4_NEAR(v1, v2, VXMATH_TOLERANCE)

#define EXPECT_VEC2_NEAR(v1, v2, tol) \
    do { \
        EXPECT_NEAR((v1).x, (v2).x, tol) << "Vx2DVector.x mismatch"; \
        EXPECT_NEAR((v1).y, (v2).y, tol) << "Vx2DVector.y mismatch"; \
    } while(0)

#define EXPECT_VEC2_EQ(v1, v2) EXPECT_VEC2_NEAR(v1, v2, VXMATH_TOLERANCE)

//=============================================================================
// Quaternion Comparison Macros
//=============================================================================

/**
 * Quaternions q and -q represent the same rotation, so we check both.
 */
inline bool QuaternionNear(const VxQuaternion& q1, const VxQuaternion& q2, float tol) {
    bool direct = (std::abs(q1.x - q2.x) < tol &&
                   std::abs(q1.y - q2.y) < tol &&
                   std::abs(q1.z - q2.z) < tol &&
                   std::abs(q1.w - q2.w) < tol);
    bool negated = (std::abs(q1.x + q2.x) < tol &&
                    std::abs(q1.y + q2.y) < tol &&
                    std::abs(q1.z + q2.z) < tol &&
                    std::abs(q1.w + q2.w) < tol);
    return direct || negated;
}

#define EXPECT_QUAT_NEAR(q1, q2, tol) \
    EXPECT_TRUE(QuaternionNear(q1, q2, tol)) \
        << "Quaternion mismatch: (" << (q1).x << "," << (q1).y << "," << (q1).z << "," << (q1).w << ") vs " \
        << "(" << (q2).x << "," << (q2).y << "," << (q2).z << "," << (q2).w << ")"

#define EXPECT_QUAT_EQ(q1, q2) EXPECT_QUAT_NEAR(q1, q2, VXMATH_TOLERANCE)

//=============================================================================
// Matrix Comparison Macros
//=============================================================================

inline bool MatrixNear(const VxMatrix& m1, const VxMatrix& m2, float tol) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (std::abs(m1[i][j] - m2[i][j]) >= tol) {
                return false;
            }
        }
    }
    return true;
}

inline std::string MatrixToString(const VxMatrix& m) {
    std::ostringstream oss;
    oss << "[\n";
    for (int i = 0; i < 4; ++i) {
        oss << "  [" << m[i][0] << ", " << m[i][1] << ", " << m[i][2] << ", " << m[i][3] << "]\n";
    }
    oss << "]";
    return oss.str();
}

#define EXPECT_MAT_NEAR(m1, m2, tol) \
    do { \
        for (int _i = 0; _i < 4; ++_i) { \
            for (int _j = 0; _j < 4; ++_j) { \
                EXPECT_NEAR((m1)[_i][_j], (m2)[_i][_j], tol) \
                    << "Matrix element [" << _i << "][" << _j << "] mismatch"; \
            } \
        } \
    } while(0)

#define EXPECT_MAT_EQ(m1, m2) EXPECT_MAT_NEAR(m1, m2, VXMATH_TOLERANCE)

//=============================================================================
// Plane Comparison Macros
//=============================================================================

#define EXPECT_PLANE_NEAR(p1, p2, tol) \
    do { \
        EXPECT_VEC3_NEAR((p1).GetNormal(), (p2).GetNormal(), tol); \
        EXPECT_NEAR((p1).m_D, (p2).m_D, tol) << "VxPlane.m_D mismatch"; \
    } while(0)

#define EXPECT_PLANE_EQ(p1, p2) EXPECT_PLANE_NEAR(p1, p2, VXMATH_TOLERANCE)

//=============================================================================
// Bbox Comparison Macros
//=============================================================================

#define EXPECT_BBOX_NEAR(b1, b2, tol) \
    do { \
        EXPECT_VEC3_NEAR((b1).Min, (b2).Min, tol); \
        EXPECT_VEC3_NEAR((b1).Max, (b2).Max, tol); \
    } while(0)

#define EXPECT_BBOX_EQ(b1, b2) EXPECT_BBOX_NEAR(b1, b2, VXMATH_TOLERANCE)

//=============================================================================
// Ray Comparison Macros
//=============================================================================

#define EXPECT_RAY_NEAR(r1, r2, tol) \
    do { \
        EXPECT_VEC3_NEAR((r1).m_Origin, (r2).m_Origin, tol); \
        EXPECT_VEC3_NEAR((r1).m_Direction, (r2).m_Direction, tol); \
    } while(0)

#define EXPECT_RAY_EQ(r1, r2) EXPECT_RAY_NEAR(r1, r2, VXMATH_TOLERANCE)

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * Create a random float in range [min, max].
 */
inline float RandomFloat(float min, float max) {
    static std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(min, max);
    return dist(gen);
}

/**
 * Create a random VxVector with components in range [min, max].
 */
inline VxVector RandomVector(float min = -100.0f, float max = 100.0f) {
    return VxVector(RandomFloat(min, max), RandomFloat(min, max), RandomFloat(min, max));
}

/**
 * Create a random unit vector.
 */
inline VxVector RandomUnitVector() {
    VxVector v = RandomVector(-1.0f, 1.0f);
    v.Normalize();
    return v;
}

/**
 * Create a random rotation matrix.
 */
inline VxMatrix RandomRotationMatrix() {
    VxMatrix mat;
    VxVector axis = RandomUnitVector();
    float angle = RandomFloat(0.0f, 2.0f * PI);
    Vx3DMatrixFromRotation(mat, axis, angle);
    return mat;
}

/**
 * Create a random transformation matrix (rotation + translation).
 */
inline VxMatrix RandomTransformMatrix() {
    VxMatrix mat = RandomRotationMatrix();
    mat[3][0] = RandomFloat(-100.0f, 100.0f);
    mat[3][1] = RandomFloat(-100.0f, 100.0f);
    mat[3][2] = RandomFloat(-100.0f, 100.0f);
    return mat;
}

/**
 * Check if a float is approximately equal to another.
 */
inline bool FloatNear(float a, float b, float tol = VXMATH_TOLERANCE) {
    return std::abs(a - b) < tol;
}

/**
 * Get magnitude of quaternion.
 */
inline float QuaternionMagnitude(const VxQuaternion& q) {
    return std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
}

/**
 * Verify quaternion is unit length.
 */
inline bool IsUnitQuaternion(const VxQuaternion& q, float tol = VXMATH_TOLERANCE) {
    return std::abs(QuaternionMagnitude(q) - 1.0f) < tol;
}

/**
 * Create identity matrix.
 */
inline VxMatrix MakeIdentityMatrix() {
    VxMatrix mat;
    mat.SetIdentity();
    return mat;
}

//=============================================================================
// Test Data Structures for Parameterized Tests
//=============================================================================

/**
 * Test case for vector operations.
 */
struct VectorTestCase {
    VxVector input;
    VxVector expected;
    const char* name;
};

/**
 * Test case for matrix operations.
 */
struct MatrixTestCase {
    VxMatrix input;
    VxMatrix expected;
    const char* name;
};

/**
 * Test case for quaternion operations.
 */
struct QuaternionTestCase {
    VxQuaternion input;
    VxQuaternion expected;
    const char* name;
};

#endif // VXMATH_TEST_HELPERS_H
