/**
 * @file SIMDQuaternionTest.cpp
 * @brief Tests for SIMD quaternion operations.
 *
 * Tests inline SIMD operations defined in VxSIMD.h/VxSIMD.inl:
 * - VxSIMDNormalizeQuaternion
 * - VxSIMDMultiplyQuaternion
 * - VxSIMDSlerpQuaternion
 */

#include <gtest/gtest.h>
#include <cmath>

#include "VxSIMD.h"
#include "SIMDTestCommon.h"

#if defined(VX_SIMD_SSE)

namespace {

using namespace SIMDTest;

class SIMDQuaternionTest : public SIMDTest::SIMDTestBase {};

//=============================================================================
// Normalize Quaternion Tests
//=============================================================================

TEST_F(SIMDQuaternionTest, Normalize_IdentityUnchanged) {
    VxQuaternion q(0.0f, 0.0f, 0.0f, 1.0f);  // Identity
    VxSIMDNormalizeQuaternion(&q);

    EXPECT_NEAR(q.x, 0.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(q.y, 0.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(q.z, 0.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(q.w, 1.0f, SIMD_SCALAR_TOL);
}

TEST_F(SIMDQuaternionTest, Normalize_ResultHasUnitLength) {
    for (int i = 0; i < 100; ++i) {
        VxQuaternion q = RandomQuaternion();
        VxSIMDNormalizeQuaternion(&q);

        float len = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        EXPECT_NEAR(len, 1.0f, SIMD_SCALAR_TOL);
    }
}

TEST_F(SIMDQuaternionTest, Normalize_MatchesScalar) {
    for (int i = 0; i < 100; ++i) {
        VxQuaternion q = RandomQuaternion();

        VxQuaternion expected = q;
        expected.Normalize();

        VxSIMDNormalizeQuaternion(&q);

        EXPECT_SIMD_QUAT_NEAR(q, expected, SIMD_SCALAR_TOL);
    }
}

//=============================================================================
// Multiply Quaternion Tests
//=============================================================================

TEST_F(SIMDQuaternionTest, Multiply_IdentityLeft) {
    VxQuaternion identity(0.0f, 0.0f, 0.0f, 1.0f);
    VxQuaternion q = RandomUnitQuaternion();
    VxQuaternion result;

    VxSIMDMultiplyQuaternion(&result, &identity, &q);

    EXPECT_SIMD_QUAT_NEAR(result, q, SIMD_SCALAR_TOL);
}

TEST_F(SIMDQuaternionTest, Multiply_IdentityRight) {
    VxQuaternion identity(0.0f, 0.0f, 0.0f, 1.0f);
    VxQuaternion q = RandomUnitQuaternion();
    VxQuaternion result;

    VxSIMDMultiplyQuaternion(&result, &q, &identity);

    EXPECT_SIMD_QUAT_NEAR(result, q, SIMD_SCALAR_TOL);
}

TEST_F(SIMDQuaternionTest, Multiply_MatchesScalar) {
    for (int i = 0; i < 100; ++i) {
        VxQuaternion a = RandomUnitQuaternion();
        VxQuaternion b = RandomUnitQuaternion();

        VxQuaternion expected = a * b;

        VxQuaternion result;
        VxSIMDMultiplyQuaternion(&result, &a, &b);

        EXPECT_SIMD_QUAT_NEAR(result, expected, SIMD_SCALAR_TOL);
    }
}

TEST_F(SIMDQuaternionTest, Multiply_AssociativeProperty) {
    VxQuaternion a = RandomUnitQuaternion();
    VxQuaternion b = RandomUnitQuaternion();
    VxQuaternion c = RandomUnitQuaternion();

    // (a * b) * c
    VxQuaternion ab, abc1;
    VxSIMDMultiplyQuaternion(&ab, &a, &b);
    VxSIMDMultiplyQuaternion(&abc1, &ab, &c);

    // a * (b * c)
    VxQuaternion bc, abc2;
    VxSIMDMultiplyQuaternion(&bc, &b, &c);
    VxSIMDMultiplyQuaternion(&abc2, &a, &bc);

    EXPECT_SIMD_QUAT_NEAR(abc1, abc2, SIMD_ACCUMULATED_TOL);
}

//=============================================================================
// Slerp Tests
//=============================================================================

TEST_F(SIMDQuaternionTest, Slerp_Boundaries) {
    VxQuaternion a = RandomUnitQuaternion();
    VxQuaternion b = RandomUnitQuaternion();
    VxQuaternion result;

    // t=0 should give a
    VxSIMDSlerpQuaternion(&result, 0.0f, &a, &b);
    EXPECT_SIMD_QUAT_NEAR(result, a, SIMD_SCALAR_TOL);

    // t=1 should give b
    VxSIMDSlerpQuaternion(&result, 1.0f, &a, &b);
    EXPECT_SIMD_QUAT_NEAR(result, b, SIMD_SCALAR_TOL);
}

TEST_F(SIMDQuaternionTest, Slerp_MidpointHasUnitLength) {
    for (int i = 0; i < 50; ++i) {
        VxQuaternion a = RandomUnitQuaternion();
        VxQuaternion b = RandomUnitQuaternion();
        VxQuaternion result;

        VxSIMDSlerpQuaternion(&result, 0.5f, &a, &b);

        float len = std::sqrt(result.x * result.x + result.y * result.y +
                              result.z * result.z + result.w * result.w);
        EXPECT_NEAR(len, 1.0f, SIMD_SCALAR_TOL);
    }
}

TEST_F(SIMDQuaternionTest, Slerp_MatchesScalar) {
    for (int i = 0; i < 50; ++i) {
        VxQuaternion a = RandomUnitQuaternion();
        VxQuaternion b = RandomUnitQuaternion();
        float t = RandomInterpolationFactor();

        // Use the free function Slerp
        VxQuaternion expected = Slerp(t, a, b);

        VxQuaternion result;
        VxSIMDSlerpQuaternion(&result, t, &a, &b);

        EXPECT_SIMD_QUAT_NEAR(result, expected, SIMD_SCALAR_TOL);
    }
}

TEST_F(SIMDQuaternionTest, Slerp_SameQuaternion) {
    VxQuaternion q = RandomUnitQuaternion();
    VxQuaternion result;

    VxSIMDSlerpQuaternion(&result, 0.5f, &q, &q);

    EXPECT_SIMD_QUAT_NEAR(result, q, SIMD_SCALAR_TOL);
}

//=============================================================================
// Edge Cases
//=============================================================================

TEST_F(SIMDQuaternionTest, EdgeCase_SlerpNearlyIdentical) {
    VxQuaternion q1 = RandomUnitQuaternion();
    VxQuaternion q2 = q1;
    // Tiny perturbation
    q2.x += 1e-7f;
    q2.Normalize();

    VxQuaternion result;
    VxSIMDSlerpQuaternion(&result, 0.5f, &q1, &q2);

    // Should not produce NaN or invalid results
    EXPECT_FALSE(std::isnan(result.x));
    EXPECT_FALSE(std::isnan(result.y));
    EXPECT_FALSE(std::isnan(result.z));
    EXPECT_FALSE(std::isnan(result.w));

    float len = std::sqrt(result.x * result.x + result.y * result.y +
                          result.z * result.z + result.w * result.w);
    EXPECT_NEAR(len, 1.0f, SIMD_SCALAR_TOL);
}

TEST_F(SIMDQuaternionTest, StressTest_ChainedMultiplications) {
    VxQuaternion result(0.0f, 0.0f, 0.0f, 1.0f);

    // Chain many small rotations
    VxVector axis(0.0f, 1.0f, 0.0f);
    float smallAngle = PI / 100.0f;

    VxQuaternion smallRot;
    smallRot.FromRotation(axis, smallAngle);

    for (int i = 0; i < 200; ++i) {
        VxQuaternion temp;
        VxSIMDMultiplyQuaternion(&temp, &result, &smallRot);
        result = temp;

        // Renormalize periodically to prevent drift
        if (i % 50 == 0) {
            VxSIMDNormalizeQuaternion(&result);
        }
    }

    // Should still be unit length
    float len = std::sqrt(result.x * result.x + result.y * result.y +
                          result.z * result.z + result.w * result.w);
    EXPECT_NEAR(len, 1.0f, SIMD_ACCUMULATED_TOL);
}

} // namespace

#endif // VX_SIMD_SSE
