/**
 * @file SIMDVectorTest.cpp
 * @brief Tests for SIMD vector operations (VxVector and VxVector4).
 *
 * Tests all inline SIMD operations defined in VxSIMD.h/VxSIMD.inl:
 * - VxSIMDAddVector, VxSIMDSubtractVector, VxSIMDScaleVector
 * - VxSIMDDotVector, VxSIMDCrossVector
 * - VxSIMDLengthVector, VxSIMDLengthSquaredVector, VxSIMDDistanceVector
 * - VxSIMDLerpVector, VxSIMDReflectVector
 * - VxSIMDMinimizeVector, VxSIMDMaximizeVector
 * - VxSIMDNormalizeVector, VxSIMDRotateVector
 * - VxSIMDAddVector4, VxSIMDSubtractVector4, VxSIMDScaleVector4
 * - VxSIMDDotVector4, VxSIMDLerpVector4
 */

#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>

#include "VxSIMD.h"
#include "SIMDTestCommon.h"

#if defined(VX_SIMD_SSE)

namespace {

using namespace SIMDTest;

class SIMDVectorTest : public SIMDTest::SIMDTestBase {};

//=============================================================================
// VxVector Add Tests
//=============================================================================

TEST_F(SIMDVectorTest, Add_BasicOperation) {
    VxVector a(1.0f, 2.0f, 3.0f);
    VxVector b(4.0f, 5.0f, 6.0f);
    VxVector result;

    VxSIMDAddVector(&result, &a, &b);

    EXPECT_FLOAT_EQ(result.x, 5.0f);
    EXPECT_FLOAT_EQ(result.y, 7.0f);
    EXPECT_FLOAT_EQ(result.z, 9.0f);
}

TEST_F(SIMDVectorTest, Add_ZeroVector) {
    VxVector a(1.0f, 2.0f, 3.0f);
    VxVector zero(0.0f, 0.0f, 0.0f);
    VxVector result;

    VxSIMDAddVector(&result, &a, &zero);

    EXPECT_SIMD_VEC3_NEAR(result, a, SIMD_EXACT_TOL);
}

TEST_F(SIMDVectorTest, Add_NegativeValues) {
    VxVector a(1.0f, -2.0f, 3.0f);
    VxVector b(-1.0f, 2.0f, -3.0f);
    VxVector result;

    VxSIMDAddVector(&result, &a, &b);

    EXPECT_NEAR(result.x, 0.0f, SIMD_EXACT_TOL);
    EXPECT_NEAR(result.y, 0.0f, SIMD_EXACT_TOL);
    EXPECT_NEAR(result.z, 0.0f, SIMD_EXACT_TOL);
}

TEST_F(SIMDVectorTest, Add_MatchesScalar) {
    for (int i = 0; i < 100; ++i) {
        VxVector a = RandomVector();
        VxVector b = RandomVector();

        VxVector expected;
        Scalar::AddVector(expected, a, b);

        VxVector result;
        VxSIMDAddVector(&result, &a, &b);

        EXPECT_SIMD_VEC3_NEAR(result, expected, SIMD_EXACT_TOL);
    }
}

//=============================================================================
// VxVector Subtract Tests
//=============================================================================

TEST_F(SIMDVectorTest, Subtract_BasicOperation) {
    VxVector a(5.0f, 7.0f, 9.0f);
    VxVector b(4.0f, 5.0f, 6.0f);
    VxVector result;

    VxSIMDSubtractVector(&result, &a, &b);

    EXPECT_FLOAT_EQ(result.x, 1.0f);
    EXPECT_FLOAT_EQ(result.y, 2.0f);
    EXPECT_FLOAT_EQ(result.z, 3.0f);
}

TEST_F(SIMDVectorTest, Subtract_SameVector) {
    VxVector a(1.0f, 2.0f, 3.0f);
    VxVector result;

    VxSIMDSubtractVector(&result, &a, &a);

    EXPECT_NEAR(result.x, 0.0f, SIMD_EXACT_TOL);
    EXPECT_NEAR(result.y, 0.0f, SIMD_EXACT_TOL);
    EXPECT_NEAR(result.z, 0.0f, SIMD_EXACT_TOL);
}

TEST_F(SIMDVectorTest, Subtract_MatchesScalar) {
    for (int i = 0; i < 100; ++i) {
        VxVector a = RandomVector();
        VxVector b = RandomVector();

        VxVector expected;
        Scalar::SubtractVector(expected, a, b);

        VxVector result;
        VxSIMDSubtractVector(&result, &a, &b);

        EXPECT_SIMD_VEC3_NEAR(result, expected, SIMD_EXACT_TOL);
    }
}

//=============================================================================
// VxVector Scale Tests
//=============================================================================

TEST_F(SIMDVectorTest, Scale_BasicOperation) {
    VxVector v(1.0f, 2.0f, 3.0f);
    VxVector result;

    VxSIMDScaleVector(&result, &v, 2.0f);

    EXPECT_FLOAT_EQ(result.x, 2.0f);
    EXPECT_FLOAT_EQ(result.y, 4.0f);
    EXPECT_FLOAT_EQ(result.z, 6.0f);
}

TEST_F(SIMDVectorTest, Scale_Zero) {
    VxVector v(1.0f, 2.0f, 3.0f);
    VxVector result;

    VxSIMDScaleVector(&result, &v, 0.0f);

    EXPECT_FLOAT_EQ(result.x, 0.0f);
    EXPECT_FLOAT_EQ(result.y, 0.0f);
    EXPECT_FLOAT_EQ(result.z, 0.0f);
}

TEST_F(SIMDVectorTest, Scale_Negative) {
    VxVector v(1.0f, 2.0f, 3.0f);
    VxVector result;

    VxSIMDScaleVector(&result, &v, -1.0f);

    EXPECT_FLOAT_EQ(result.x, -1.0f);
    EXPECT_FLOAT_EQ(result.y, -2.0f);
    EXPECT_FLOAT_EQ(result.z, -3.0f);
}

TEST_F(SIMDVectorTest, Scale_MatchesScalar) {
    for (int i = 0; i < 100; ++i) {
        VxVector v = RandomVector();
        float s = RandomFloat(-10.0f, 10.0f);

        VxVector expected;
        Scalar::ScaleVector(expected, v, s);

        VxVector result;
        VxSIMDScaleVector(&result, &v, s);

        EXPECT_SIMD_VEC3_NEAR(result, expected, SIMD_EXACT_TOL);
    }
}

//=============================================================================
// VxVector Dot Product Tests
//=============================================================================

TEST_F(SIMDVectorTest, Dot_Orthogonal) {
    VxVector x(1.0f, 0.0f, 0.0f);
    VxVector y(0.0f, 1.0f, 0.0f);

    float dot = VxSIMDDotVector(&x, &y);

    EXPECT_NEAR(dot, 0.0f, SIMD_EXACT_TOL);
}

TEST_F(SIMDVectorTest, Dot_Parallel) {
    VxVector a(1.0f, 0.0f, 0.0f);
    VxVector b(2.0f, 0.0f, 0.0f);

    float dot = VxSIMDDotVector(&a, &b);

    EXPECT_NEAR(dot, 2.0f, SIMD_EXACT_TOL);
}

TEST_F(SIMDVectorTest, Dot_AntiParallel) {
    VxVector a(1.0f, 0.0f, 0.0f);
    VxVector b(-2.0f, 0.0f, 0.0f);

    float dot = VxSIMDDotVector(&a, &b);

    EXPECT_NEAR(dot, -2.0f, SIMD_EXACT_TOL);
}

TEST_F(SIMDVectorTest, Dot_MatchesScalar) {
    for (int i = 0; i < 100; ++i) {
        VxVector a = RandomVector();
        VxVector b = RandomVector();

        float expected = Scalar::DotProduct3(a, b);
        float result = VxSIMDDotVector(&a, &b);

        EXPECT_SIMD_NEAR(result, expected, SIMD_SCALAR_TOL);
    }
}

//=============================================================================
// VxVector Cross Product Tests
//=============================================================================

TEST_F(SIMDVectorTest, Cross_UnitVectors) {
    VxVector x(1.0f, 0.0f, 0.0f);
    VxVector y(0.0f, 1.0f, 0.0f);
    VxVector result;

    VxSIMDCrossVector(&result, &x, &y);

    // X × Y = Z
    EXPECT_NEAR(result.x, 0.0f, SIMD_EXACT_TOL);
    EXPECT_NEAR(result.y, 0.0f, SIMD_EXACT_TOL);
    EXPECT_NEAR(result.z, 1.0f, SIMD_EXACT_TOL);
}

TEST_F(SIMDVectorTest, Cross_AntiCommutative) {
    VxVector a = RandomVector();
    VxVector b = RandomVector();

    VxVector ab, ba;
    VxSIMDCrossVector(&ab, &a, &b);
    VxSIMDCrossVector(&ba, &b, &a);

    // a × b = -(b × a)
    EXPECT_SIMD_NEAR(ab.x, -ba.x, SIMD_SCALAR_TOL);
    EXPECT_SIMD_NEAR(ab.y, -ba.y, SIMD_SCALAR_TOL);
    EXPECT_SIMD_NEAR(ab.z, -ba.z, SIMD_SCALAR_TOL);
}

TEST_F(SIMDVectorTest, Cross_SelfIsZero) {
    VxVector v = RandomVector();
    VxVector result;

    VxSIMDCrossVector(&result, &v, &v);

    EXPECT_NEAR(result.x, 0.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(result.y, 0.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(result.z, 0.0f, SIMD_SCALAR_TOL);
}

TEST_F(SIMDVectorTest, Cross_MatchesScalar) {
    for (int i = 0; i < 100; ++i) {
        VxVector a = RandomVector();
        VxVector b = RandomVector();

        VxVector expected;
        Scalar::CrossProduct(expected, a, b);

        VxVector result;
        VxSIMDCrossVector(&result, &a, &b);

        EXPECT_SIMD_VEC3_NEAR(result, expected, SIMD_SCALAR_TOL);
    }
}

//=============================================================================
// VxVector Length Tests
//=============================================================================

TEST_F(SIMDVectorTest, Length_UnitVectors) {
    VxVector x(1.0f, 0.0f, 0.0f);
    VxVector y(0.0f, 1.0f, 0.0f);
    VxVector z(0.0f, 0.0f, 1.0f);

    EXPECT_NEAR(VxSIMDLengthVector(&x), 1.0f, SIMD_EXACT_TOL);
    EXPECT_NEAR(VxSIMDLengthVector(&y), 1.0f, SIMD_EXACT_TOL);
    EXPECT_NEAR(VxSIMDLengthVector(&z), 1.0f, SIMD_EXACT_TOL);
}

TEST_F(SIMDVectorTest, Length_ZeroVector) {
    VxVector zero(0.0f, 0.0f, 0.0f);

    float len = VxSIMDLengthVector(&zero);

    EXPECT_NEAR(len, 0.0f, SIMD_EXACT_TOL);
}

TEST_F(SIMDVectorTest, Length_MatchesScalar) {
    for (int i = 0; i < 100; ++i) {
        VxVector v = RandomVector();

        float expected = Scalar::VectorLength(v);
        float result = VxSIMDLengthVector(&v);

        EXPECT_SIMD_NEAR(result, expected, SIMD_SCALAR_TOL);
    }
}

TEST_F(SIMDVectorTest, LengthSquared_MatchesScalar) {
    for (int i = 0; i < 100; ++i) {
        VxVector v = RandomVector();

        float expected = Scalar::VectorLengthSquared(v);
        float result = VxSIMDLengthSquaredVector(&v);

        EXPECT_SIMD_NEAR(result, expected, SIMD_SCALAR_TOL);
    }
}

//=============================================================================
// VxVector Distance Tests
//=============================================================================

TEST_F(SIMDVectorTest, Distance_SamePoint) {
    VxVector v = RandomVector();

    float dist = VxSIMDDistanceVector(&v, &v);

    EXPECT_NEAR(dist, 0.0f, SIMD_EXACT_TOL);
}

TEST_F(SIMDVectorTest, Distance_KnownValues) {
    VxVector a(0.0f, 0.0f, 0.0f);
    VxVector b(3.0f, 4.0f, 0.0f);  // 3-4-5 triangle

    float dist = VxSIMDDistanceVector(&a, &b);

    EXPECT_NEAR(dist, 5.0f, SIMD_SCALAR_TOL);
}

TEST_F(SIMDVectorTest, Distance_Symmetric) {
    VxVector a = RandomVector();
    VxVector b = RandomVector();

    float ab = VxSIMDDistanceVector(&a, &b);
    float ba = VxSIMDDistanceVector(&b, &a);

    EXPECT_SIMD_NEAR(ab, ba, SIMD_EXACT_TOL);
}

TEST_F(SIMDVectorTest, Distance_MatchesScalar) {
    for (int i = 0; i < 100; ++i) {
        VxVector a = RandomVector();
        VxVector b = RandomVector();

        float expected = Scalar::VectorDistance(a, b);
        float result = VxSIMDDistanceVector(&a, &b);

        EXPECT_SIMD_NEAR(result, expected, SIMD_SCALAR_TOL);
    }
}

//=============================================================================
// VxVector Lerp Tests
//=============================================================================

TEST_F(SIMDVectorTest, Lerp_Boundaries) {
    VxVector a(0.0f, 0.0f, 0.0f);
    VxVector b(1.0f, 2.0f, 3.0f);
    VxVector result;

    // t=0 should return a
    VxSIMDLerpVector(&result, &a, &b, 0.0f);
    EXPECT_SIMD_VEC3_NEAR(result, a, SIMD_EXACT_TOL);

    // t=1 should return b
    VxSIMDLerpVector(&result, &a, &b, 1.0f);
    EXPECT_SIMD_VEC3_NEAR(result, b, SIMD_EXACT_TOL);
}

TEST_F(SIMDVectorTest, Lerp_Midpoint) {
    VxVector a(0.0f, 0.0f, 0.0f);
    VxVector b(2.0f, 4.0f, 6.0f);
    VxVector result;

    VxSIMDLerpVector(&result, &a, &b, 0.5f);

    EXPECT_NEAR(result.x, 1.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(result.y, 2.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(result.z, 3.0f, SIMD_SCALAR_TOL);
}

TEST_F(SIMDVectorTest, Lerp_MatchesScalar) {
    for (int i = 0; i < 100; ++i) {
        VxVector a = RandomVector();
        VxVector b = RandomVector();
        float t = RandomInterpolationFactor();

        VxVector expected;
        Scalar::VectorLerp(expected, a, b, t);

        VxVector result;
        VxSIMDLerpVector(&result, &a, &b, t);

        EXPECT_SIMD_VEC3_NEAR(result, expected, SIMD_SCALAR_TOL);
    }
}

//=============================================================================
// VxVector Reflect Tests
//=============================================================================

TEST_F(SIMDVectorTest, Reflect_PerpendicularIncidence) {
    // Ray straight down, normal pointing up
    VxVector incident(0.0f, -1.0f, 0.0f);
    VxVector normal(0.0f, 1.0f, 0.0f);
    VxVector result;

    VxSIMDReflectVector(&result, &incident, &normal);

    // Should reflect straight up
    EXPECT_NEAR(result.x, 0.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(result.y, 1.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(result.z, 0.0f, SIMD_SCALAR_TOL);
}

TEST_F(SIMDVectorTest, Reflect_45DegreeAngle) {
    // 45 degree incidence
    VxVector incident(1.0f, -1.0f, 0.0f);
    incident.Normalize();
    VxVector normal(0.0f, 1.0f, 0.0f);
    VxVector result;

    VxSIMDReflectVector(&result, &incident, &normal);

    // Should reflect at 45 degrees opposite
    VxVector expected(incident.x, -incident.y, incident.z);
    EXPECT_SIMD_VEC3_NEAR(result, expected, SIMD_SCALAR_TOL);
}

TEST_F(SIMDVectorTest, Reflect_MatchesScalar) {
    for (int i = 0; i < 100; ++i) {
        VxVector incident = RandomVector();
        VxVector normal = RandomUnitVector();

        VxVector expected;
        Scalar::VectorReflect(expected, incident, normal);

        VxVector result;
        VxSIMDReflectVector(&result, &incident, &normal);

        EXPECT_SIMD_VEC3_NEAR(result, expected, SIMD_SCALAR_TOL);
    }
}

//=============================================================================
// VxVector Minimize/Maximize Tests
//=============================================================================

TEST_F(SIMDVectorTest, Minimize_BasicOperation) {
    VxVector a(1.0f, 5.0f, 3.0f);
    VxVector b(4.0f, 2.0f, 6.0f);
    VxVector result;

    VxSIMDMinimizeVector(&result, &a, &b);

    EXPECT_FLOAT_EQ(result.x, 1.0f);
    EXPECT_FLOAT_EQ(result.y, 2.0f);
    EXPECT_FLOAT_EQ(result.z, 3.0f);
}

TEST_F(SIMDVectorTest, Maximize_BasicOperation) {
    VxVector a(1.0f, 5.0f, 3.0f);
    VxVector b(4.0f, 2.0f, 6.0f);
    VxVector result;

    VxSIMDMaximizeVector(&result, &a, &b);

    EXPECT_FLOAT_EQ(result.x, 4.0f);
    EXPECT_FLOAT_EQ(result.y, 5.0f);
    EXPECT_FLOAT_EQ(result.z, 6.0f);
}

TEST_F(SIMDVectorTest, MinMaximize_MatchesScalar) {
    for (int i = 0; i < 100; ++i) {
        VxVector a = RandomVector();
        VxVector b = RandomVector();

        VxVector expectedMin, expectedMax;
        Scalar::VectorMinimize(expectedMin, a, b);
        Scalar::VectorMaximize(expectedMax, a, b);

        VxVector resultMin, resultMax;
        VxSIMDMinimizeVector(&resultMin, &a, &b);
        VxSIMDMaximizeVector(&resultMax, &a, &b);

        EXPECT_SIMD_VEC3_NEAR(resultMin, expectedMin, SIMD_EXACT_TOL);
        EXPECT_SIMD_VEC3_NEAR(resultMax, expectedMax, SIMD_EXACT_TOL);
    }
}

//=============================================================================
// VxVector Normalize Tests
//=============================================================================

TEST_F(SIMDVectorTest, Normalize_UnitVectorUnchanged) {
    VxVector v(1.0f, 0.0f, 0.0f);
    VxVector original = v;

    VxSIMDNormalizeVector(&v);

    EXPECT_SIMD_VEC3_NEAR(v, original, SIMD_SCALAR_TOL);
}

TEST_F(SIMDVectorTest, Normalize_ResultHasUnitLength) {
    for (int i = 0; i < 100; ++i) {
        VxVector v = RandomVector();
        if (v.SquareMagnitude() < EPSILON) continue;

        VxSIMDNormalizeVector(&v);

        float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        EXPECT_NEAR(len, 1.0f, SIMD_SCALAR_TOL);
    }
}

TEST_F(SIMDVectorTest, Normalize_PreservesDirection) {
    for (int i = 0; i < 100; ++i) {
        VxVector v = RandomVector();
        if (v.SquareMagnitude() < EPSILON) continue;

        VxVector expected = v;
        Scalar::NormalizeVector(expected);

        VxSIMDNormalizeVector(&v);

        EXPECT_SIMD_VEC3_NEAR(v, expected, SIMD_SCALAR_TOL);
    }
}

TEST_F(SIMDVectorTest, Normalize_ZeroVectorSafe) {
    VxVector zero(0.0f, 0.0f, 0.0f);

    VxSIMDNormalizeVector(&zero);

    // Should not crash or produce NaN
    EXPECT_FALSE(std::isnan(zero.x));
    EXPECT_FALSE(std::isnan(zero.y));
    EXPECT_FALSE(std::isnan(zero.z));
}

TEST_F(SIMDVectorTest, Normalize_VerySmallVectorSafe) {
    VxVector tiny(1e-38f, 1e-38f, 1e-38f);

    VxSIMDNormalizeVector(&tiny);

    // Should not crash or produce NaN
    EXPECT_FALSE(std::isnan(tiny.x));
    EXPECT_FALSE(std::isnan(tiny.y));
    EXPECT_FALSE(std::isnan(tiny.z));
}

//=============================================================================
// VxVector Rotate Tests
//=============================================================================

TEST_F(SIMDVectorTest, Rotate_IdentityMatrix) {
    VxMatrix identity;
    identity.SetIdentity();
    VxVector v(1.0f, 2.0f, 3.0f);
    VxVector result;

    VxSIMDRotateVector(&result, &identity, &v);

    EXPECT_SIMD_VEC3_NEAR(result, v, SIMD_SCALAR_TOL);
}

TEST_F(SIMDVectorTest, Rotate_90DegreesAroundZ) {
    VxMatrix rot;
    Vx3DMatrixFromRotation(rot, VxVector(0.0f, 0.0f, 1.0f), PI / 2.0f);

    VxVector v(1.0f, 0.0f, 0.0f);
    VxVector result;

    VxSIMDRotateVector(&result, &rot, &v);

    // X axis rotated 90 degrees around Z should become Y
    EXPECT_NEAR(result.x, 0.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(result.y, 1.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(result.z, 0.0f, SIMD_SCALAR_TOL);
}

TEST_F(SIMDVectorTest, Rotate_PreservesLength) {
    for (int i = 0; i < 50; ++i) {
        VxMatrix rot = RandomRotationMatrix();
        VxVector v = RandomVector();

        float originalLen = v.Magnitude();

        VxVector result;
        VxSIMDRotateVector(&result, &rot, &v);

        float newLen = result.Magnitude();
        EXPECT_SIMD_NEAR(newLen, originalLen, SIMD_SCALAR_TOL);
    }
}

TEST_F(SIMDVectorTest, Rotate_MatchesScalar) {
    for (int i = 0; i < 100; ++i) {
        VxMatrix rot = RandomRotationMatrix();
        VxVector v = RandomVector();

        VxVector expected;
        Vx3DRotateVector(&expected, rot, &v);

        VxVector result;
        VxSIMDRotateVector(&result, &rot, &v);

        EXPECT_SIMD_VEC3_NEAR(result, expected, SIMD_SCALAR_TOL);
    }
}

//=============================================================================
// VxVector4 Tests
//=============================================================================

TEST_F(SIMDVectorTest, Vector4_Add_MatchesScalar) {
    for (int i = 0; i < 100; ++i) {
        VxVector4 a = RandomVector4();
        VxVector4 b = RandomVector4();

        VxVector4 expected;
        Scalar::AddVector4(expected, a, b);

        VxVector4 result;
        VxSIMDAddVector4(&result, &a, &b);

        EXPECT_SIMD_VEC4_NEAR(result, expected, SIMD_EXACT_TOL);
    }
}

TEST_F(SIMDVectorTest, Vector4_Subtract_MatchesScalar) {
    for (int i = 0; i < 100; ++i) {
        VxVector4 a = RandomVector4();
        VxVector4 b = RandomVector4();

        VxVector4 expected;
        Scalar::SubtractVector4(expected, a, b);

        VxVector4 result;
        VxSIMDSubtractVector4(&result, &a, &b);

        EXPECT_SIMD_VEC4_NEAR(result, expected, SIMD_EXACT_TOL);
    }
}

TEST_F(SIMDVectorTest, Vector4_Scale_MatchesScalar) {
    for (int i = 0; i < 100; ++i) {
        VxVector4 v = RandomVector4();
        float s = RandomFloat(-10.0f, 10.0f);

        VxVector4 expected;
        Scalar::ScaleVector4(expected, v, s);

        VxVector4 result;
        VxSIMDScaleVector4(&result, &v, s);

        EXPECT_SIMD_VEC4_NEAR(result, expected, SIMD_EXACT_TOL);
    }
}

TEST_F(SIMDVectorTest, Vector4_Dot_MatchesScalar) {
    for (int i = 0; i < 100; ++i) {
        VxVector4 a = RandomVector4();
        VxVector4 b = RandomVector4();

        float expected = Scalar::DotProduct4(a, b);
        float result = VxSIMDDotVector4(&a, &b);

        EXPECT_SIMD_NEAR(result, expected, SIMD_SCALAR_TOL);
    }
}

TEST_F(SIMDVectorTest, Vector4_Lerp_Boundaries) {
    VxVector4 a(0.0f, 0.0f, 0.0f, 0.0f);
    VxVector4 b(1.0f, 2.0f, 3.0f, 4.0f);
    VxVector4 result;

    // t=0
    VxSIMDLerpVector4(&result, &a, &b, 0.0f);
    EXPECT_SIMD_VEC4_NEAR(result, a, SIMD_EXACT_TOL);

    // t=1
    VxSIMDLerpVector4(&result, &a, &b, 1.0f);
    EXPECT_SIMD_VEC4_NEAR(result, b, SIMD_EXACT_TOL);
}

TEST_F(SIMDVectorTest, Vector4_Lerp_MatchesScalar) {
    for (int i = 0; i < 100; ++i) {
        VxVector4 a = RandomVector4();
        VxVector4 b = RandomVector4();
        float t = RandomInterpolationFactor();

        VxVector4 expected;
        Scalar::LerpVector4(expected, a, b, t);

        VxVector4 result;
        VxSIMDLerpVector4(&result, &a, &b, t);

        EXPECT_SIMD_VEC4_NEAR(result, expected, SIMD_SCALAR_TOL);
    }
}

//=============================================================================
// Edge Cases and Stress Tests
//=============================================================================

TEST_F(SIMDVectorTest, EdgeCase_LargeValues) {
    VxVector a(1e6f, 1e6f, 1e6f);
    VxVector b(1e6f, 1e6f, 1e6f);
    VxVector result;

    VxSIMDAddVector(&result, &a, &b);

    EXPECT_NEAR(result.x, 2e6f, 1e3f);
    EXPECT_NEAR(result.y, 2e6f, 1e3f);
    EXPECT_NEAR(result.z, 2e6f, 1e3f);
}

TEST_F(SIMDVectorTest, EdgeCase_SmallValues) {
    VxVector a(1e-6f, 1e-6f, 1e-6f);
    VxVector b(1e-6f, 1e-6f, 1e-6f);
    VxVector result;

    VxSIMDAddVector(&result, &a, &b);

    EXPECT_NEAR(result.x, 2e-6f, 1e-10f);
    EXPECT_NEAR(result.y, 2e-6f, 1e-10f);
    EXPECT_NEAR(result.z, 2e-6f, 1e-10f);
}

TEST_F(SIMDVectorTest, EdgeCase_MixedSigns) {
    VxVector a(1.0f, -2.0f, 3.0f);
    VxVector b(-4.0f, 5.0f, -6.0f);
    VxVector result;

    VxSIMDAddVector(&result, &a, &b);

    EXPECT_FLOAT_EQ(result.x, -3.0f);
    EXPECT_FLOAT_EQ(result.y, 3.0f);
    EXPECT_FLOAT_EQ(result.z, -3.0f);
}

TEST_F(SIMDVectorTest, StressTest_ManyOperations) {
    // Stress test with many sequential operations
    VxVector v(1.0f, 0.0f, 0.0f);

    for (int i = 0; i < 1000; ++i) {
        VxVector temp;
        VxSIMDScaleVector(&temp, &v, 1.001f);
        VxSIMDNormalizeVector(&temp);
        v = temp;
    }

    // Should still be unit length
    float len = v.Magnitude();
    EXPECT_NEAR(len, 1.0f, SIMD_ACCUMULATED_TOL);
}

} // namespace

#endif // VX_SIMD_SSE
