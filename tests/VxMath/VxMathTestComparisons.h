/**
 * @file VxMathTestComparisons.h
 * @brief Type-specific comparison macros and functions for VxMath tests.
 *
 * This header provides comparison utilities for all VxMath types, including
 * GTest assertion helpers with detailed error messages.
 *
 * Consolidates helpers from:
 * - SIMDDispatchConsistencyTest.cpp (ExpectNear* functions)
 * - VectorTest.cpp (VectorsApproxEqual)
 * - QuaternionTest.cpp (QuaternionsApproxEqual)
 * - FrustumTest.cpp (EXPECT_VECTORS_NEAR, EXPECT_PLANES_NEAR)
 */

#ifndef VXMATH_TEST_COMPARISONS_H
#define VXMATH_TEST_COMPARISONS_H

#include <gtest/gtest.h>
#include <cmath>
#include <sstream>
#include <string>

#include "VxMath.h"
#include "VxMathTestTolerances.h"

namespace VxMathTest {

//=============================================================================
// String Formatting Utilities
//=============================================================================

inline std::string VectorToString(const VxVector& v) {
    std::ostringstream oss;
    oss << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return oss.str();
}

inline std::string Vector4ToString(const VxVector4& v) {
    std::ostringstream oss;
    oss << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
    return oss.str();
}

inline std::string Vector2DToString(const Vx2DVector& v) {
    std::ostringstream oss;
    oss << "(" << v.x << ", " << v.y << ")";
    return oss.str();
}

inline std::string QuaternionToString(const VxQuaternion& q) {
    std::ostringstream oss;
    oss << "(" << q.x << ", " << q.y << ", " << q.z << ", " << q.w << ")";
    return oss.str();
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

inline std::string PlaneToString(const VxPlane& p) {
    std::ostringstream oss;
    oss << "Plane(n=" << VectorToString(p.m_Normal) << ", d=" << p.m_D << ")";
    return oss.str();
}

inline std::string BboxToString(const VxBbox& b) {
    std::ostringstream oss;
    oss << "Bbox(min=" << VectorToString(b.Min) << ", max=" << VectorToString(b.Max) << ")";
    return oss.str();
}

inline std::string RayToString(const VxRay& r) {
    std::ostringstream oss;
    oss << "Ray(origin=" << VectorToString(r.m_Origin) << ", dir=" << VectorToString(r.m_Direction) << ")";
    return oss.str();
}

inline std::string RectToString(const VxRect& r) {
    std::ostringstream oss;
    oss << "Rect(l=" << r.left << ", t=" << r.top << ", r=" << r.right << ", b=" << r.bottom << ")";
    return oss.str();
}

inline std::string SphereToString(const VxSphere& s) {
    std::ostringstream oss;
    oss << "Sphere(center=" << VectorToString(s.Center()) << ", r=" << s.Radius() << ")";
    return oss.str();
}

//=============================================================================
// Boolean Comparison Functions
//=============================================================================

/// Compare two VxVector values with tolerance
inline bool VectorNearBool(const VxVector& a, const VxVector& b, float tol = STANDARD_TOL) {
    return ScaleRelativeNear(a.x, b.x, tol) &&
           ScaleRelativeNear(a.y, b.y, tol) &&
           ScaleRelativeNear(a.z, b.z, tol);
}

/// Compare two VxVector4 values with tolerance
inline bool Vector4NearBool(const VxVector4& a, const VxVector4& b, float tol = STANDARD_TOL) {
    return ScaleRelativeNear(a.x, b.x, tol) &&
           ScaleRelativeNear(a.y, b.y, tol) &&
           ScaleRelativeNear(a.z, b.z, tol) &&
           ScaleRelativeNear(a.w, b.w, tol);
}

/// Compare two Vx2DVector values with tolerance
inline bool Vector2DNearBool(const Vx2DVector& a, const Vx2DVector& b, float tol = STANDARD_TOL) {
    return ScaleRelativeNear(a.x, b.x, tol) &&
           ScaleRelativeNear(a.y, b.y, tol);
}

/// Compare two VxQuaternion values with tolerance (handles q == -q equivalence)
inline bool QuaternionNearBool(const VxQuaternion& a, const VxQuaternion& b, float tol = STANDARD_TOL) {
    // Direct comparison
    bool direct = ScaleRelativeNear(a.x, b.x, tol) &&
                  ScaleRelativeNear(a.y, b.y, tol) &&
                  ScaleRelativeNear(a.z, b.z, tol) &&
                  ScaleRelativeNear(a.w, b.w, tol);
    if (direct) return true;

    // Negated comparison (q and -q represent the same rotation)
    return ScaleRelativeNear(a.x, -b.x, tol) &&
           ScaleRelativeNear(a.y, -b.y, tol) &&
           ScaleRelativeNear(a.z, -b.z, tol) &&
           ScaleRelativeNear(a.w, -b.w, tol);
}

/// Compare two VxMatrix values with tolerance
inline bool MatrixNearBool(const VxMatrix& a, const VxMatrix& b, float tol = STANDARD_TOL) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (!ScaleRelativeNear(a[i][j], b[i][j], tol)) {
                return false;
            }
        }
    }
    return true;
}

/// Compare two VxPlane values with tolerance
inline bool PlaneNearBool(const VxPlane& a, const VxPlane& b, float tol = STANDARD_TOL) {
    return VectorNearBool(a.m_Normal, b.m_Normal, tol) &&
           ScaleRelativeNear(a.m_D, b.m_D, tol);
}

/// Compare two VxBbox values with tolerance
inline bool BboxNearBool(const VxBbox& a, const VxBbox& b, float tol = STANDARD_TOL) {
    return VectorNearBool(a.Min, b.Min, tol) &&
           VectorNearBool(a.Max, b.Max, tol);
}

/// Compare two VxRay values with tolerance
inline bool RayNearBool(const VxRay& a, const VxRay& b, float tol = STANDARD_TOL) {
    return VectorNearBool(a.m_Origin, b.m_Origin, tol) &&
           VectorNearBool(a.m_Direction, b.m_Direction, tol);
}

/// Compare two VxRect values with tolerance
inline bool RectNearBool(const VxRect& a, const VxRect& b, float tol = STANDARD_TOL) {
    return ScaleRelativeNear(a.left, b.left, tol) &&
           ScaleRelativeNear(a.top, b.top, tol) &&
           ScaleRelativeNear(a.right, b.right, tol) &&
           ScaleRelativeNear(a.bottom, b.bottom, tol);
}

/// Compare two VxSphere values with tolerance
inline bool SphereNearBool(const VxSphere& a, const VxSphere& b, float tol = STANDARD_TOL) {
    return VectorNearBool(a.Center(), b.Center(), tol) &&
           ScaleRelativeNear(a.Radius(), b.Radius(), tol);
}

//=============================================================================
// GTest Assertion Functions (provide rich error messages)
//=============================================================================

/// GTest assertion for VxVector comparison
inline ::testing::AssertionResult VectorNear(const VxVector& actual, const VxVector& expected, float tol = STANDARD_TOL) {
    if (VectorNearBool(actual, expected, tol)) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
        << "VxVector mismatch:\n"
        << "  Expected: " << VectorToString(expected) << "\n"
        << "  Actual:   " << VectorToString(actual) << "\n"
        << "  Tolerance: " << tol;
}

/// GTest assertion for VxVector4 comparison
inline ::testing::AssertionResult Vector4Near(const VxVector4& actual, const VxVector4& expected, float tol = STANDARD_TOL) {
    if (Vector4NearBool(actual, expected, tol)) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
        << "VxVector4 mismatch:\n"
        << "  Expected: " << Vector4ToString(expected) << "\n"
        << "  Actual:   " << Vector4ToString(actual) << "\n"
        << "  Tolerance: " << tol;
}

/// GTest assertion for VxQuaternion comparison
inline ::testing::AssertionResult QuaternionNear(const VxQuaternion& actual, const VxQuaternion& expected, float tol = STANDARD_TOL) {
    if (QuaternionNearBool(actual, expected, tol)) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
        << "VxQuaternion mismatch (q and -q both checked):\n"
        << "  Expected: " << QuaternionToString(expected) << "\n"
        << "  Actual:   " << QuaternionToString(actual) << "\n"
        << "  Tolerance: " << tol;
}

/// GTest assertion for VxMatrix comparison
inline ::testing::AssertionResult MatrixNear(const VxMatrix& actual, const VxMatrix& expected, float tol = STANDARD_TOL) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (!ScaleRelativeNear(actual[i][j], expected[i][j], tol)) {
                return ::testing::AssertionFailure()
                    << "VxMatrix mismatch at [" << i << "][" << j << "]:\n"
                    << "  Expected: " << expected[i][j] << "\n"
                    << "  Actual:   " << actual[i][j] << "\n"
                    << "  Tolerance: " << tol << "\n"
                    << "  Full expected matrix:\n" << MatrixToString(expected) << "\n"
                    << "  Full actual matrix:\n" << MatrixToString(actual);
            }
        }
    }
    return ::testing::AssertionSuccess();
}

/// GTest assertion for VxPlane comparison
inline ::testing::AssertionResult PlaneNear(const VxPlane& actual, const VxPlane& expected, float tol = STANDARD_TOL) {
    if (PlaneNearBool(actual, expected, tol)) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
        << "VxPlane mismatch:\n"
        << "  Expected: " << PlaneToString(expected) << "\n"
        << "  Actual:   " << PlaneToString(actual) << "\n"
        << "  Tolerance: " << tol;
}

/// GTest assertion for VxBbox comparison
inline ::testing::AssertionResult BboxNear(const VxBbox& actual, const VxBbox& expected, float tol = STANDARD_TOL) {
    if (BboxNearBool(actual, expected, tol)) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
        << "VxBbox mismatch:\n"
        << "  Expected: " << BboxToString(expected) << "\n"
        << "  Actual:   " << BboxToString(actual) << "\n"
        << "  Tolerance: " << tol;
}

/// GTest assertion for VxRay comparison
inline ::testing::AssertionResult RayNear(const VxRay& actual, const VxRay& expected, float tol = STANDARD_TOL) {
    if (RayNearBool(actual, expected, tol)) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
        << "VxRay mismatch:\n"
        << "  Expected: " << RayToString(expected) << "\n"
        << "  Actual:   " << RayToString(actual) << "\n"
        << "  Tolerance: " << tol;
}

/// GTest assertion for VxRect comparison
inline ::testing::AssertionResult RectNear(const VxRect& actual, const VxRect& expected, float tol = STANDARD_TOL) {
    if (RectNearBool(actual, expected, tol)) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
        << "VxRect mismatch:\n"
        << "  Expected: " << RectToString(expected) << "\n"
        << "  Actual:   " << RectToString(actual) << "\n"
        << "  Tolerance: " << tol;
}

/// GTest assertion for VxSphere comparison
inline ::testing::AssertionResult SphereNear(const VxSphere& actual, const VxSphere& expected, float tol = STANDARD_TOL) {
    if (SphereNearBool(actual, expected, tol)) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure()
        << "VxSphere mismatch:\n"
        << "  Expected: " << SphereToString(expected) << "\n"
        << "  Actual:   " << SphereToString(actual) << "\n"
        << "  Tolerance: " << tol;
}

//=============================================================================
// Inline Expect Functions (for use in test files like SIMDDispatchConsistencyTest)
//=============================================================================

inline void ExpectNearVec3(const VxVector& a, const VxVector& b, float tol = STANDARD_TOL) {
    EXPECT_TRUE(ScaleRelativeNear(a.x, b.x, tol)) << "x a=" << a.x << " b=" << b.x << " tol=" << tol;
    EXPECT_TRUE(ScaleRelativeNear(a.y, b.y, tol)) << "y a=" << a.y << " b=" << b.y << " tol=" << tol;
    EXPECT_TRUE(ScaleRelativeNear(a.z, b.z, tol)) << "z a=" << a.z << " b=" << b.z << " tol=" << tol;
}

inline void ExpectNearVec4(const VxVector4& a, const VxVector4& b, float tol = STANDARD_TOL) {
    EXPECT_TRUE(ScaleRelativeNear(a.x, b.x, tol)) << "x a=" << a.x << " b=" << b.x << " tol=" << tol;
    EXPECT_TRUE(ScaleRelativeNear(a.y, b.y, tol)) << "y a=" << a.y << " b=" << b.y << " tol=" << tol;
    EXPECT_TRUE(ScaleRelativeNear(a.z, b.z, tol)) << "z a=" << a.z << " b=" << b.z << " tol=" << tol;
    EXPECT_TRUE(ScaleRelativeNear(a.w, b.w, tol)) << "w a=" << a.w << " b=" << b.w << " tol=" << tol;
}

inline void ExpectNearQuat(const VxQuaternion& a, const VxQuaternion& b, float tol = STANDARD_TOL) {
    EXPECT_TRUE(QuaternionNearBool(a, b, tol))
        << "Quaternion mismatch: " << QuaternionToString(a) << " vs " << QuaternionToString(b);
}

inline void ExpectNearMatrix(const VxMatrix& a, const VxMatrix& b, float tol = STANDARD_TOL) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_TRUE(ScaleRelativeNear(a[i][j], b[i][j], tol))
                << "m[" << i << "][" << j << "] a=" << a[i][j] << " b=" << b[i][j];
        }
    }
}

inline void ExpectNearPlane(const VxPlane& a, const VxPlane& b, float tol = STANDARD_TOL) {
    ExpectNearVec3(a.m_Normal, b.m_Normal, tol);
    EXPECT_TRUE(ScaleRelativeNear(a.m_D, b.m_D, tol)) << "d a=" << a.m_D << " b=" << b.m_D;
}

inline void ExpectNearBbox(const VxBbox& a, const VxBbox& b, float tol = STANDARD_TOL) {
    ExpectNearVec3(a.Min, b.Min, tol);
    ExpectNearVec3(a.Max, b.Max, tol);
}

inline void ExpectNearRect(const VxRect& a, const VxRect& b, float tol = STANDARD_TOL) {
    EXPECT_TRUE(ScaleRelativeNear(a.left, b.left, tol)) << "left a=" << a.left << " b=" << b.left << " tol=" << tol;
    EXPECT_TRUE(ScaleRelativeNear(a.top, b.top, tol)) << "top a=" << a.top << " b=" << b.top << " tol=" << tol;
    EXPECT_TRUE(ScaleRelativeNear(a.right, b.right, tol)) << "right a=" << a.right << " b=" << b.right << " tol=" << tol;
    EXPECT_TRUE(ScaleRelativeNear(a.bottom, b.bottom, tol)) << "bottom a=" << a.bottom << " b=" << b.bottom << " tol=" << tol;
}

inline void ExpectNear4(const float* a, const float* b, float tol = STANDARD_TOL) {
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(ScaleRelativeNear(a[i], b[i], tol)) << "idx=" << i << " a=" << a[i] << " b=" << b[i];
    }
}

} // namespace VxMathTest

//=============================================================================
// GTest Assertion Macros (backward-compatible with existing tests)
//=============================================================================

// Vector macros with scale-relative tolerance
#define EXPECT_VEC3_NEAR_REL(v1, v2, tol) \
    EXPECT_TRUE(VxMathTest::VectorNear(v1, v2, tol))

#define EXPECT_VEC3_EQ_REL(v1, v2) \
    EXPECT_TRUE(VxMathTest::VectorNear(v1, v2, VxMathTest::STANDARD_TOL))

#define EXPECT_VEC4_NEAR_REL(v1, v2, tol) \
    EXPECT_TRUE(VxMathTest::Vector4Near(v1, v2, tol))

#define EXPECT_VEC4_EQ_REL(v1, v2) \
    EXPECT_TRUE(VxMathTest::Vector4Near(v1, v2, VxMathTest::STANDARD_TOL))

// Quaternion macros
#define EXPECT_QUAT_NEAR_REL(q1, q2, tol) \
    EXPECT_TRUE(VxMathTest::QuaternionNear(q1, q2, tol))

#define EXPECT_QUAT_EQ_REL(q1, q2) \
    EXPECT_TRUE(VxMathTest::QuaternionNear(q1, q2, VxMathTest::STANDARD_TOL))

// Matrix macros
#define EXPECT_MAT_NEAR_REL(m1, m2, tol) \
    EXPECT_TRUE(VxMathTest::MatrixNear(m1, m2, tol))

#define EXPECT_MAT_EQ_REL(m1, m2) \
    EXPECT_TRUE(VxMathTest::MatrixNear(m1, m2, VxMathTest::STANDARD_TOL))

// Geometric type macros
#define EXPECT_PLANE_NEAR_REL(p1, p2, tol) \
    EXPECT_TRUE(VxMathTest::PlaneNear(p1, p2, tol))

#define EXPECT_BBOX_NEAR_REL(b1, b2, tol) \
    EXPECT_TRUE(VxMathTest::BboxNear(b1, b2, tol))

#define EXPECT_RAY_NEAR_REL(r1, r2, tol) \
    EXPECT_TRUE(VxMathTest::RayNear(r1, r2, tol))

#define EXPECT_RECT_NEAR_REL(r1, r2, tol) \
    EXPECT_TRUE(VxMathTest::RectNear(r1, r2, tol))

#define EXPECT_SPHERE_NEAR(s1, s2, tol) \
    EXPECT_TRUE(VxMathTest::SphereNear(s1, s2, tol))

#endif // VXMATH_TEST_COMPARISONS_H
