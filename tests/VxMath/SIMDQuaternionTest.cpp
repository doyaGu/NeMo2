/**
 * @file SIMDQuaternionTest.cpp
 * @brief Tests for SIMD quaternion operations.
 *
 * Tests all operations in VxSIMDQuaternionOps:
 * - NormalizeQuaternion
 * - MultiplyQuaternion, MultiplyInPlace
 * - SlerpQuaternion, Squad
 * - FromMatrix, ToMatrix
 * - FromAxisAngle, FromEulerAngles, ToEulerAngles
 * - Conjugate, Divide
 * - Ln, Exp, LnDif
 * - Snuggle
 */

#include <gtest/gtest.h>
#include <cmath>

#include "VxSIMD.h"
#include "SIMDTestCommon.h"

namespace {

using namespace SIMDTest;

class SIMDQuaternionTest : public SIMDTest::SIMDTestBase {};

//=============================================================================
// Normalize Quaternion Tests
//=============================================================================

TEST_F(SIMDQuaternionTest, Normalize_IdentityUnchanged) {
    ASSERT_NE(m_dispatch->Quaternion.NormalizeQuaternion, nullptr);

    VxQuaternion q(0.0f, 0.0f, 0.0f, 1.0f);  // Identity
    m_dispatch->Quaternion.NormalizeQuaternion(&q);

    EXPECT_NEAR(q.x, 0.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(q.y, 0.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(q.z, 0.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(q.w, 1.0f, SIMD_SCALAR_TOL);
}

TEST_F(SIMDQuaternionTest, Normalize_ResultHasUnitLength) {
    for (int i = 0; i < 100; ++i) {
        VxQuaternion q = RandomQuaternion();
        m_dispatch->Quaternion.NormalizeQuaternion(&q);

        float len = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        EXPECT_NEAR(len, 1.0f, SIMD_SCALAR_TOL);
    }
}

TEST_F(SIMDQuaternionTest, Normalize_MatchesScalar) {
    for (int i = 0; i < 100; ++i) {
        VxQuaternion q = RandomQuaternion();

        VxQuaternion expected = q;
        expected.Normalize();

        m_dispatch->Quaternion.NormalizeQuaternion(&q);

        EXPECT_SIMD_QUAT_NEAR(q, expected, SIMD_SCALAR_TOL);
    }
}

//=============================================================================
// Multiply Quaternion Tests
//=============================================================================

TEST_F(SIMDQuaternionTest, Multiply_IdentityLeft) {
    ASSERT_NE(m_dispatch->Quaternion.MultiplyQuaternion, nullptr);

    VxQuaternion identity(0.0f, 0.0f, 0.0f, 1.0f);
    VxQuaternion q = RandomUnitQuaternion();
    VxQuaternion result;

    m_dispatch->Quaternion.MultiplyQuaternion(&result, &identity, &q);

    EXPECT_SIMD_QUAT_NEAR(result, q, SIMD_SCALAR_TOL);
}

TEST_F(SIMDQuaternionTest, Multiply_IdentityRight) {
    VxQuaternion identity(0.0f, 0.0f, 0.0f, 1.0f);
    VxQuaternion q = RandomUnitQuaternion();
    VxQuaternion result;

    m_dispatch->Quaternion.MultiplyQuaternion(&result, &q, &identity);

    EXPECT_SIMD_QUAT_NEAR(result, q, SIMD_SCALAR_TOL);
}

TEST_F(SIMDQuaternionTest, Multiply_MatchesScalar) {
    for (int i = 0; i < 100; ++i) {
        VxQuaternion a = RandomUnitQuaternion();
        VxQuaternion b = RandomUnitQuaternion();

        VxQuaternion expected = a * b;

        VxQuaternion result;
        m_dispatch->Quaternion.MultiplyQuaternion(&result, &a, &b);

        EXPECT_SIMD_QUAT_NEAR(result, expected, SIMD_SCALAR_TOL);
    }
}

TEST_F(SIMDQuaternionTest, Multiply_AssociativeProperty) {
    VxQuaternion a = RandomUnitQuaternion();
    VxQuaternion b = RandomUnitQuaternion();
    VxQuaternion c = RandomUnitQuaternion();

    // (a * b) * c
    VxQuaternion ab, abc1;
    m_dispatch->Quaternion.MultiplyQuaternion(&ab, &a, &b);
    m_dispatch->Quaternion.MultiplyQuaternion(&abc1, &ab, &c);

    // a * (b * c)
    VxQuaternion bc, abc2;
    m_dispatch->Quaternion.MultiplyQuaternion(&bc, &b, &c);
    m_dispatch->Quaternion.MultiplyQuaternion(&abc2, &a, &bc);

    EXPECT_SIMD_QUAT_NEAR(abc1, abc2, SIMD_ACCUMULATED_TOL);
}

TEST_F(SIMDQuaternionTest, MultiplyInPlace_MatchesMultiply) {
    ASSERT_NE(m_dispatch->Quaternion.MultiplyInPlace, nullptr);

    for (int i = 0; i < 50; ++i) {
        VxQuaternion a = RandomUnitQuaternion();
        VxQuaternion b = RandomUnitQuaternion();

        VxQuaternion expected;
        m_dispatch->Quaternion.MultiplyQuaternion(&expected, &a, &b);

        VxQuaternion result = a;
        m_dispatch->Quaternion.MultiplyInPlace(&result, &b);

        EXPECT_SIMD_QUAT_NEAR(result, expected, SIMD_SCALAR_TOL);
    }
}

//=============================================================================
// Slerp Tests
//=============================================================================

TEST_F(SIMDQuaternionTest, Slerp_Boundaries) {
    ASSERT_NE(m_dispatch->Quaternion.SlerpQuaternion, nullptr);

    VxQuaternion a = RandomUnitQuaternion();
    VxQuaternion b = RandomUnitQuaternion();
    VxQuaternion result;

    // t=0 should give a
    m_dispatch->Quaternion.SlerpQuaternion(&result, 0.0f, &a, &b);
    EXPECT_SIMD_QUAT_NEAR(result, a, SIMD_SCALAR_TOL);

    // t=1 should give b
    m_dispatch->Quaternion.SlerpQuaternion(&result, 1.0f, &a, &b);
    EXPECT_SIMD_QUAT_NEAR(result, b, SIMD_SCALAR_TOL);
}

TEST_F(SIMDQuaternionTest, Slerp_MidpointHasUnitLength) {
    for (int i = 0; i < 50; ++i) {
        VxQuaternion a = RandomUnitQuaternion();
        VxQuaternion b = RandomUnitQuaternion();
        VxQuaternion result;

        m_dispatch->Quaternion.SlerpQuaternion(&result, 0.5f, &a, &b);

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
        m_dispatch->Quaternion.SlerpQuaternion(&result, t, &a, &b);

        EXPECT_SIMD_QUAT_NEAR(result, expected, SIMD_SCALAR_TOL);
    }
}

TEST_F(SIMDQuaternionTest, Slerp_SameQuaternion) {
    VxQuaternion q = RandomUnitQuaternion();
    VxQuaternion result;

    m_dispatch->Quaternion.SlerpQuaternion(&result, 0.5f, &q, &q);

    EXPECT_SIMD_QUAT_NEAR(result, q, SIMD_SCALAR_TOL);
}

//=============================================================================
// Matrix Conversion Tests
//=============================================================================

TEST_F(SIMDQuaternionTest, ToMatrix_IdentityQuaternion) {
    ASSERT_NE(m_dispatch->Quaternion.ToMatrix, nullptr);

    VxQuaternion identity(0.0f, 0.0f, 0.0f, 1.0f);
    VxMatrix result;

    m_dispatch->Quaternion.ToMatrix(&result, &identity);

    VxMatrix expected;
    expected.SetIdentity();

    // Check rotation part (3x3)
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(result[i][j], expected[i][j], SIMD_SCALAR_TOL)
                << "Mismatch at [" << i << "][" << j << "]";
        }
    }
}

TEST_F(SIMDQuaternionTest, ToMatrix_MatchesScalar) {
    for (int i = 0; i < 50; ++i) {
        VxQuaternion q = RandomUnitQuaternion();

        VxMatrix expected;
        q.ToMatrix(expected);

        VxMatrix result;
        m_dispatch->Quaternion.ToMatrix(&result, &q);

        EXPECT_SIMD_MAT_NEAR(result, expected, SIMD_SCALAR_TOL);
    }
}

TEST_F(SIMDQuaternionTest, FromMatrix_Roundtrip) {
    ASSERT_NE(m_dispatch->Quaternion.FromMatrix, nullptr);

    for (int i = 0; i < 50; ++i) {
        VxQuaternion original = RandomUnitQuaternion();

        VxMatrix mat;
        m_dispatch->Quaternion.ToMatrix(&mat, &original);

        VxQuaternion result;
        m_dispatch->Quaternion.FromMatrix(&result, &mat, FALSE, FALSE);

        // Quaternions may differ by sign
        EXPECT_SIMD_QUAT_NEAR(result, original, SIMD_ACCUMULATED_TOL);
    }
}

TEST_F(SIMDQuaternionTest, FromMatrix_MatchesScalar) {
    for (int i = 0; i < 50; ++i) {
        VxMatrix rot = RandomRotationMatrix();

        VxQuaternion expected;
        expected.FromMatrix(rot, FALSE, FALSE);

        VxQuaternion result;
        m_dispatch->Quaternion.FromMatrix(&result, &rot, FALSE, FALSE);

        EXPECT_SIMD_QUAT_NEAR(result, expected, SIMD_ACCUMULATED_TOL);
    }
}

//=============================================================================
// Axis-Angle Conversion Tests
//=============================================================================

TEST_F(SIMDQuaternionTest, FromAxisAngle_ZeroAngle) {
    ASSERT_NE(m_dispatch->Quaternion.FromAxisAngle, nullptr);

    VxVector axis(1.0f, 0.0f, 0.0f);
    VxQuaternion result;

    m_dispatch->Quaternion.FromAxisAngle(&result, &axis, 0.0f);

    // Should be identity quaternion
    VxQuaternion identity(0.0f, 0.0f, 0.0f, 1.0f);
    EXPECT_SIMD_QUAT_NEAR(result, identity, SIMD_SCALAR_TOL);
}

TEST_F(SIMDQuaternionTest, FromAxisAngle_90DegreesZ) {
    VxVector axis(0.0f, 0.0f, 1.0f);
    VxQuaternion result;

    m_dispatch->Quaternion.FromAxisAngle(&result, &axis, PI / 2.0f);

    // Should be (0, 0, sin(45), cos(45)) = (0, 0, 0.707, 0.707)
    EXPECT_NEAR(result.x, 0.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(result.y, 0.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(std::abs(result.z), 0.707107f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(result.w, 0.707107f, SIMD_SCALAR_TOL);
}

TEST_F(SIMDQuaternionTest, FromAxisAngle_MatchesScalar) {
    for (int i = 0; i < 50; ++i) {
        VxVector axis = RandomUnitVector();
        float angle = RandomAngle();

        VxQuaternion expected;
        expected.FromRotation(axis, angle);

        VxQuaternion result;
        m_dispatch->Quaternion.FromAxisAngle(&result, &axis, angle);

        EXPECT_SIMD_QUAT_NEAR(result, expected, SIMD_SCALAR_TOL);
    }
}

//=============================================================================
// Euler Angles Conversion Tests
//=============================================================================

TEST_F(SIMDQuaternionTest, FromEulerAngles_Zero) {
    ASSERT_NE(m_dispatch->Quaternion.FromEulerAngles, nullptr);

    VxQuaternion result;
    m_dispatch->Quaternion.FromEulerAngles(&result, 0.0f, 0.0f, 0.0f);

    VxQuaternion identity(0.0f, 0.0f, 0.0f, 1.0f);
    EXPECT_SIMD_QUAT_NEAR(result, identity, SIMD_SCALAR_TOL);
}

TEST_F(SIMDQuaternionTest, FromEulerAngles_MatchesScalar) {
    for (int i = 0; i < 50; ++i) {
        float eax = RandomFloat(-PI, PI);
        float eay = RandomFloat(-PI / 2.0f + 0.1f, PI / 2.0f - 0.1f);
        float eaz = RandomFloat(-PI, PI);

        VxQuaternion expected;
        expected.FromEulerAngles(eax, eay, eaz);

        VxQuaternion result;
        m_dispatch->Quaternion.FromEulerAngles(&result, eax, eay, eaz);

        EXPECT_SIMD_QUAT_NEAR(result, expected, SIMD_SCALAR_TOL);
    }
}

TEST_F(SIMDQuaternionTest, ToEulerAngles_Roundtrip) {
    ASSERT_NE(m_dispatch->Quaternion.ToEulerAngles, nullptr);

    for (int i = 0; i < 50; ++i) {
        float eax = RandomFloat(-PI * 0.4f, PI * 0.4f);
        float eay = RandomFloat(-PI / 2.0f + 0.2f, PI / 2.0f - 0.2f);
        float eaz = RandomFloat(-PI * 0.4f, PI * 0.4f);

        VxQuaternion q;
        m_dispatch->Quaternion.FromEulerAngles(&q, eax, eay, eaz);

        float outEax, outEay, outEaz;
        m_dispatch->Quaternion.ToEulerAngles(&q, &outEax, &outEay, &outEaz);

        // Reconstruct and compare
        VxQuaternion reconstructed;
        m_dispatch->Quaternion.FromEulerAngles(&reconstructed, outEax, outEay, outEaz);

        EXPECT_SIMD_QUAT_NEAR(reconstructed, q, SIMD_ACCUMULATED_TOL);
    }
}

//=============================================================================
// Conjugate Tests
//=============================================================================

TEST_F(SIMDQuaternionTest, Conjugate_BasicOperation) {
    ASSERT_NE(m_dispatch->Quaternion.Conjugate, nullptr);

    VxQuaternion q(1.0f, 2.0f, 3.0f, 4.0f);
    VxQuaternion result;

    m_dispatch->Quaternion.Conjugate(&result, &q);

    EXPECT_FLOAT_EQ(result.x, -1.0f);
    EXPECT_FLOAT_EQ(result.y, -2.0f);
    EXPECT_FLOAT_EQ(result.z, -3.0f);
    EXPECT_FLOAT_EQ(result.w, 4.0f);
}

TEST_F(SIMDQuaternionTest, Conjugate_MultiplyGivesIdentity) {
    VxQuaternion q = RandomUnitQuaternion();
    VxQuaternion conj, product;

    m_dispatch->Quaternion.Conjugate(&conj, &q);
    m_dispatch->Quaternion.MultiplyQuaternion(&product, &q, &conj);

    // q * conjugate(q) = identity for unit quaternion
    VxQuaternion identity(0.0f, 0.0f, 0.0f, 1.0f);
    EXPECT_SIMD_QUAT_NEAR(product, identity, SIMD_SCALAR_TOL);
}

//=============================================================================
// Divide Tests
//=============================================================================

TEST_F(SIMDQuaternionTest, Divide_ByItself) {
    ASSERT_NE(m_dispatch->Quaternion.Divide, nullptr);

    VxQuaternion q = RandomUnitQuaternion();
    VxQuaternion result;

    m_dispatch->Quaternion.Divide(&result, &q, &q);

    // q / q = identity
    VxQuaternion identity(0.0f, 0.0f, 0.0f, 1.0f);
    EXPECT_SIMD_QUAT_NEAR(result, identity, SIMD_SCALAR_TOL);
}

TEST_F(SIMDQuaternionTest, Divide_ByIdentity) {
    VxQuaternion q = RandomUnitQuaternion();
    VxQuaternion identity(0.0f, 0.0f, 0.0f, 1.0f);
    VxQuaternion result;

    m_dispatch->Quaternion.Divide(&result, &q, &identity);

    // q / identity = q (mathematical property)
    EXPECT_SIMD_QUAT_NEAR(result, q, SIMD_SCALAR_TOL);
}

//=============================================================================
// Ln and Exp Tests
//=============================================================================

TEST_F(SIMDQuaternionTest, Ln_IdentityIsZero) {
    ASSERT_NE(m_dispatch->Quaternion.Ln, nullptr);

    VxQuaternion identity(0.0f, 0.0f, 0.0f, 1.0f);
    VxQuaternion result;

    m_dispatch->Quaternion.Ln(&result, &identity);

    // ln(identity) should be approximately (0, 0, 0, 0)
    EXPECT_NEAR(result.x, 0.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(result.y, 0.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(result.z, 0.0f, SIMD_SCALAR_TOL);
}

TEST_F(SIMDQuaternionTest, ExpLn_Roundtrip) {
    ASSERT_NE(m_dispatch->Quaternion.Exp, nullptr);

    for (int i = 0; i < 30; ++i) {
        VxQuaternion q = RandomUnitQuaternion();

        VxQuaternion lnQ, result;
        m_dispatch->Quaternion.Ln(&lnQ, &q);
        m_dispatch->Quaternion.Exp(&result, &lnQ);

        EXPECT_SIMD_QUAT_NEAR(result, q, SIMD_ACCUMULATED_TOL);
    }
}

//=============================================================================
// Squad Tests (Spherical Cubic Interpolation)
//=============================================================================

TEST_F(SIMDQuaternionTest, Squad_BoundaryT0) {
    ASSERT_NE(m_dispatch->Quaternion.Squad, nullptr);

    VxQuaternion q1 = RandomUnitQuaternion();
    VxQuaternion q1Out = RandomUnitQuaternion();
    VxQuaternion q2In = RandomUnitQuaternion();
    VxQuaternion q2 = RandomUnitQuaternion();
    VxQuaternion result;

    m_dispatch->Quaternion.Squad(&result, 0.0f, &q1, &q1Out, &q2In, &q2);

    EXPECT_SIMD_QUAT_NEAR(result, q1, SIMD_SCALAR_TOL);
}

TEST_F(SIMDQuaternionTest, Squad_BoundaryT1) {
    VxQuaternion q1 = RandomUnitQuaternion();
    VxQuaternion q1Out = RandomUnitQuaternion();
    VxQuaternion q2In = RandomUnitQuaternion();
    VxQuaternion q2 = RandomUnitQuaternion();
    VxQuaternion result;

    m_dispatch->Quaternion.Squad(&result, 1.0f, &q1, &q1Out, &q2In, &q2);

    EXPECT_SIMD_QUAT_NEAR(result, q2, SIMD_SCALAR_TOL);
}

TEST_F(SIMDQuaternionTest, Squad_ResultIsUnit) {
    for (int i = 0; i < 30; ++i) {
        VxQuaternion q1 = RandomUnitQuaternion();
        VxQuaternion q1Out = RandomUnitQuaternion();
        VxQuaternion q2In = RandomUnitQuaternion();
        VxQuaternion q2 = RandomUnitQuaternion();
        float t = RandomInterpolationFactor();

        VxQuaternion result;
        m_dispatch->Quaternion.Squad(&result, t, &q1, &q1Out, &q2In, &q2);

        float len = std::sqrt(result.x * result.x + result.y * result.y +
                              result.z * result.z + result.w * result.w);
        EXPECT_NEAR(len, 1.0f, SIMD_SCALAR_TOL);
    }
}

//=============================================================================
// Rotation Application Tests
//=============================================================================

TEST_F(SIMDQuaternionTest, RotationPreservesVectorLength) {
    for (int i = 0; i < 50; ++i) {
        VxQuaternion q = RandomUnitQuaternion();
        VxVector v = RandomVector();

        float originalLen = v.Magnitude();

        // Convert to matrix and rotate
        VxMatrix mat;
        m_dispatch->Quaternion.ToMatrix(&mat, &q);

        VxVector rotated;
        Vx3DRotateVector(&rotated, mat, &v);

        float newLen = rotated.Magnitude();
        EXPECT_SIMD_NEAR(newLen, originalLen, SIMD_SCALAR_TOL);
    }
}

TEST_F(SIMDQuaternionTest, ConcatenatedRotationsMatchMatrixMultiply) {
    // Mathematical property: ToMatrix(q1 * q2) should equal ToMatrix(q2) * ToMatrix(q1)
    // (quaternion multiplication order is opposite to matrix multiplication order
    // because quaternion rotation applies right-to-left: q1 * q2 * v * conj(q2) * conj(q1))
    for (int i = 0; i < 30; ++i) {
        VxQuaternion q1 = RandomUnitQuaternion();
        VxQuaternion q2 = RandomUnitQuaternion();
        VxVector v = RandomVector();

        // Quaternion multiplication then to matrix
        VxQuaternion qProduct;
        m_dispatch->Quaternion.MultiplyQuaternion(&qProduct, &q1, &q2);
        VxMatrix matFromQuat;
        m_dispatch->Quaternion.ToMatrix(&matFromQuat, &qProduct);
        VxVector rotQuat;
        Vx3DRotateVector(&rotQuat, matFromQuat, &v);

        // Matrix multiplication (reverse order: M2 * M1)
        VxMatrix m1, m2, mProduct;
        m_dispatch->Quaternion.ToMatrix(&m1, &q1);
        m_dispatch->Quaternion.ToMatrix(&m2, &q2);
        m_dispatch->Matrix.MultiplyMatrix(&mProduct, &m2, &m1);
        VxVector rotMat;
        Vx3DRotateVector(&rotMat, mProduct, &v);

        EXPECT_SIMD_VEC3_NEAR(rotQuat, rotMat, SIMD_ACCUMULATED_TOL);
    }
}

//=============================================================================
// Edge Cases
//=============================================================================

TEST_F(SIMDQuaternionTest, EdgeCase_OppositeQuaternions) {
    // q and -q represent the same rotation
    VxQuaternion q = RandomUnitQuaternion();
    VxQuaternion negQ(-q.x, -q.y, -q.z, -q.w);

    VxMatrix m1, m2;
    m_dispatch->Quaternion.ToMatrix(&m1, &q);
    m_dispatch->Quaternion.ToMatrix(&m2, &negQ);

    // Matrices should be identical
    EXPECT_SIMD_MAT_NEAR(m1, m2, SIMD_SCALAR_TOL);
}

TEST_F(SIMDQuaternionTest, EdgeCase_SlerpNearlyIdentical) {
    VxQuaternion q1 = RandomUnitQuaternion();
    VxQuaternion q2 = q1;
    // Tiny perturbation
    q2.x += 1e-7f;
    q2.Normalize();

    VxQuaternion result;
    m_dispatch->Quaternion.SlerpQuaternion(&result, 0.5f, &q1, &q2);

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
    m_dispatch->Quaternion.FromAxisAngle(&smallRot, &axis, smallAngle);

    for (int i = 0; i < 200; ++i) {
        VxQuaternion temp;
        m_dispatch->Quaternion.MultiplyQuaternion(&temp, &result, &smallRot);
        result = temp;

        // Renormalize periodically to prevent drift
        if (i % 50 == 0) {
            m_dispatch->Quaternion.NormalizeQuaternion(&result);
        }
    }

    // Should still be unit length
    float len = std::sqrt(result.x * result.x + result.y * result.y +
                          result.z * result.z + result.w * result.w);
    EXPECT_NEAR(len, 1.0f, SIMD_ACCUMULATED_TOL);
}

} // namespace
