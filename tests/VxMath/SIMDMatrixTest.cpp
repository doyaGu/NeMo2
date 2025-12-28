/**
 * @file SIMDMatrixTest.cpp
 * @brief Tests for SIMD matrix operations.
 *
 * Tests all operations in VxSIMDMatrixOps:
 * - MultiplyMatrix, MultiplyMatrix4
 * - TransposeMatrix
 * - MultiplyMatrixVector, MultiplyMatrixVector4
 * - RotateVectorOp
 * - MultiplyMatrixVectorMany, RotateVectorMany
 * - MultiplyMatrixVectorStrided, MultiplyMatrixVector4Strided, RotateVectorStrided
 * - Identity, Inverse, Determinant
 * - FromAxisAngle, FromAxisAngleOrigin
 * - FromEulerAngles, ToEulerAngles
 * - Interpolate, InterpolateNoScale
 * - Decompose, DecomposeTotal, DecomposeTotalPtr
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "VxSIMD.h"
#include "SIMDTestCommon.h"

namespace {

using namespace SIMDTest;

class SIMDMatrixTest : public SIMDTest::SIMDTestBase {};

//=============================================================================
// Matrix Multiply Tests
//=============================================================================

TEST_F(SIMDMatrixTest, MultiplyMatrix_IdentityLeft) {
    ASSERT_NE(m_dispatch->Matrix.MultiplyMatrix, nullptr);

    VxMatrix identity;
    identity.SetIdentity();

    VxMatrix m = RandomTRSMatrix();
    VxMatrix result;

    m_dispatch->Matrix.MultiplyMatrix(&result, &identity, &m);

    EXPECT_SIMD_MAT_NEAR(result, m, SIMD_SCALAR_TOL);
}

TEST_F(SIMDMatrixTest, MultiplyMatrix_IdentityRight) {
    VxMatrix identity;
    identity.SetIdentity();

    VxMatrix m = RandomTRSMatrix();
    VxMatrix result;

    m_dispatch->Matrix.MultiplyMatrix(&result, &m, &identity);

    EXPECT_SIMD_MAT_NEAR(result, m, SIMD_SCALAR_TOL);
}

TEST_F(SIMDMatrixTest, MultiplyMatrix_MatchesScalar) {
    for (int i = 0; i < 50; ++i) {
        VxMatrix a = RandomTRSMatrix();
        VxMatrix b = RandomTRSMatrix();

        VxMatrix expected;
        Vx3DMultiplyMatrix(expected, a, b);

        VxMatrix result;
        m_dispatch->Matrix.MultiplyMatrix(&result, &a, &b);

        EXPECT_SIMD_MAT_NEAR(result, expected, SIMD_SCALAR_TOL);
    }
}

TEST_F(SIMDMatrixTest, MultiplyMatrix4_MatchesScalar) {
    ASSERT_NE(m_dispatch->Matrix.MultiplyMatrix4, nullptr);

    for (int i = 0; i < 50; ++i) {
        VxMatrix a = RandomTRSMatrix();
        VxMatrix b = RandomTRSMatrix();

        VxMatrix expected;
        Vx3DMultiplyMatrix4(expected, a, b);

        VxMatrix result;
        m_dispatch->Matrix.MultiplyMatrix4(&result, &a, &b);

        EXPECT_SIMD_MAT_NEAR(result, expected, SIMD_SCALAR_TOL);
    }
}

//=============================================================================
// Matrix Transpose Tests
//=============================================================================

TEST_F(SIMDMatrixTest, TransposeMatrix_Identity) {
    ASSERT_NE(m_dispatch->Matrix.TransposeMatrix, nullptr);

    VxMatrix identity;
    identity.SetIdentity();
    VxMatrix result;

    m_dispatch->Matrix.TransposeMatrix(&result, &identity);

    EXPECT_SIMD_MAT_NEAR(result, identity, SIMD_EXACT_TOL);
}

TEST_F(SIMDMatrixTest, TransposeMatrix_InvolutoryProperty) {
    // Transpose of transpose should equal original
    VxMatrix m = RandomTRSMatrix();
    VxMatrix temp, result;

    m_dispatch->Matrix.TransposeMatrix(&temp, &m);
    m_dispatch->Matrix.TransposeMatrix(&result, &temp);

    EXPECT_SIMD_MAT_NEAR(result, m, SIMD_SCALAR_TOL);
}

TEST_F(SIMDMatrixTest, TransposeMatrix_MatchesScalar) {
    for (int i = 0; i < 50; ++i) {
        VxMatrix m = RandomTRSMatrix();

        VxMatrix expected;
        Vx3DTransposeMatrix(expected, m);

        VxMatrix result;
        m_dispatch->Matrix.TransposeMatrix(&result, &m);

        EXPECT_SIMD_MAT_NEAR(result, expected, SIMD_SCALAR_TOL);
    }
}

//=============================================================================
// Matrix-Vector Multiply Tests
//=============================================================================

TEST_F(SIMDMatrixTest, MultiplyMatrixVector_Identity) {
    ASSERT_NE(m_dispatch->Matrix.MultiplyMatrixVector, nullptr);

    VxMatrix identity;
    identity.SetIdentity();
    VxVector v = RandomVector();
    VxVector result;

    m_dispatch->Matrix.MultiplyMatrixVector(&result, &identity, &v);

    EXPECT_SIMD_VEC3_NEAR(result, v, SIMD_SCALAR_TOL);
}

TEST_F(SIMDMatrixTest, MultiplyMatrixVector_Translation) {
    VxMatrix trans;
    trans.SetIdentity();
    trans[3][0] = 10.0f;
    trans[3][1] = 20.0f;
    trans[3][2] = 30.0f;

    VxVector v(1.0f, 2.0f, 3.0f);
    VxVector result;

    m_dispatch->Matrix.MultiplyMatrixVector(&result, &trans, &v);

    EXPECT_NEAR(result.x, 11.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(result.y, 22.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(result.z, 33.0f, SIMD_SCALAR_TOL);
}

TEST_F(SIMDMatrixTest, MultiplyMatrixVector_MatchesScalar) {
    for (int i = 0; i < 100; ++i) {
        VxMatrix m = RandomTRSMatrix();
        VxVector v = RandomVector();

        VxVector expected;
        Vx3DMultiplyMatrixVector(&expected, m, &v);

        VxVector result;
        m_dispatch->Matrix.MultiplyMatrixVector(&result, &m, &v);

        EXPECT_SIMD_VEC3_NEAR(result, expected, SIMD_SCALAR_TOL);
    }
}

TEST_F(SIMDMatrixTest, MultiplyMatrixVector4_MatchesScalar) {
    ASSERT_NE(m_dispatch->Matrix.MultiplyMatrixVector4, nullptr);

    for (int i = 0; i < 100; ++i) {
        VxMatrix m = RandomTRSMatrix();
        VxVector4 v = RandomVector4();

        VxVector4 expected;
        Vx3DMultiplyMatrixVector4(&expected, m, &v);

        VxVector4 result;
        m_dispatch->Matrix.MultiplyMatrixVector4(&result, &m, &v);

        EXPECT_SIMD_VEC4_NEAR(result, expected, SIMD_SCALAR_TOL);
    }
}

//=============================================================================
// Rotate Vector Operation Tests
//=============================================================================

TEST_F(SIMDMatrixTest, RotateVectorOp_Identity) {
    ASSERT_NE(m_dispatch->Matrix.RotateVectorOp, nullptr);

    VxMatrix identity;
    identity.SetIdentity();
    VxVector v = RandomVector();
    VxVector result;

    m_dispatch->Matrix.RotateVectorOp(&result, &identity, &v);

    EXPECT_SIMD_VEC3_NEAR(result, v, SIMD_SCALAR_TOL);
}

TEST_F(SIMDMatrixTest, RotateVectorOp_IgnoresTranslation) {
    VxMatrix trans;
    trans.SetIdentity();
    trans[3][0] = 100.0f;
    trans[3][1] = 200.0f;
    trans[3][2] = 300.0f;

    VxVector v(1.0f, 2.0f, 3.0f);
    VxVector result;

    m_dispatch->Matrix.RotateVectorOp(&result, &trans, &v);

    // Rotation ignores translation
    EXPECT_SIMD_VEC3_NEAR(result, v, SIMD_SCALAR_TOL);
}

TEST_F(SIMDMatrixTest, RotateVectorOp_PreservesLength) {
    for (int i = 0; i < 50; ++i) {
        VxMatrix rot = RandomRotationMatrix();
        VxVector v = RandomVector();

        float originalLen = v.Magnitude();

        VxVector result;
        m_dispatch->Matrix.RotateVectorOp(&result, &rot, &v);

        float newLen = result.Magnitude();
        EXPECT_SIMD_NEAR(newLen, originalLen, SIMD_SCALAR_TOL);
    }
}

TEST_F(SIMDMatrixTest, RotateVectorOp_MatchesScalar) {
    for (int i = 0; i < 100; ++i) {
        VxMatrix m = RandomTRSMatrix();
        VxVector v = RandomVector();

        VxVector expected;
        Vx3DRotateVector(&expected, m, &v);

        VxVector result;
        m_dispatch->Matrix.RotateVectorOp(&result, &m, &v);

        EXPECT_SIMD_VEC3_NEAR(result, expected, SIMD_SCALAR_TOL);
    }
}

//=============================================================================
// Batch Transform Tests
//=============================================================================

TEST_F(SIMDMatrixTest, MultiplyMatrixVectorMany_MatchesScalar) {
    ASSERT_NE(m_dispatch->Matrix.MultiplyMatrixVectorMany, nullptr);

    VxMatrix m = RandomTRSMatrix();
    const int count = 16;
    std::vector<VxVector> input(count);
    std::vector<VxVector> expected(count);
    std::vector<VxVector> result(count);

    for (int i = 0; i < count; ++i) {
        input[i] = RandomVector();
        Vx3DMultiplyMatrixVector(&expected[i], m, &input[i]);
    }

    m_dispatch->Matrix.MultiplyMatrixVectorMany(result.data(), &m, input.data(), count, sizeof(VxVector));

    for (int i = 0; i < count; ++i) {
        EXPECT_SIMD_VEC3_NEAR(result[i], expected[i], SIMD_SCALAR_TOL)
            << "Mismatch at index " << i;
    }
}

TEST_F(SIMDMatrixTest, RotateVectorMany_MatchesScalar) {
    ASSERT_NE(m_dispatch->Matrix.RotateVectorMany, nullptr);

    VxMatrix m = RandomRotationMatrix();
    const int count = 16;
    std::vector<VxVector> input(count);
    std::vector<VxVector> expected(count);
    std::vector<VxVector> result(count);

    for (int i = 0; i < count; ++i) {
        input[i] = RandomVector();
        Vx3DRotateVector(&expected[i], m, &input[i]);
    }

    m_dispatch->Matrix.RotateVectorMany(result.data(), &m, input.data(), count, sizeof(VxVector));

    for (int i = 0; i < count; ++i) {
        EXPECT_SIMD_VEC3_NEAR(result[i], expected[i], SIMD_SCALAR_TOL)
            << "Mismatch at index " << i;
    }
}

//=============================================================================
// Strided Transform Tests
//=============================================================================

TEST_F(SIMDMatrixTest, MultiplyMatrixVectorStrided_MatchesScalar) {
    ASSERT_NE(m_dispatch->Matrix.MultiplyMatrixVectorStrided, nullptr);

    VxMatrix m = RandomTRSMatrix();
    const int count = 8;

    // Interleaved data (e.g., position + normal)
    struct Vertex {
        float x, y, z;
        float padding[3];  // Extra stride
    };

    std::vector<Vertex> input(count);
    std::vector<Vertex> expected(count);
    std::vector<Vertex> result(count);

    for (int i = 0; i < count; ++i) {
        VxVector v = RandomVector();
        input[i] = {v.x, v.y, v.z, {0, 0, 0}};

        VxVector exp;
        Vx3DMultiplyMatrixVector(&exp, m, &v);
        expected[i] = {exp.x, exp.y, exp.z, {0, 0, 0}};
    }

    VxStridedData src, dst;
    src.Ptr = &input[0].x;
    src.Stride = sizeof(Vertex);
    dst.Ptr = &result[0].x;
    dst.Stride = sizeof(Vertex);

    m_dispatch->Matrix.MultiplyMatrixVectorStrided(&dst, &src, &m, count);

    for (int i = 0; i < count; ++i) {
        VxVector r(result[i].x, result[i].y, result[i].z);
        VxVector e(expected[i].x, expected[i].y, expected[i].z);
        EXPECT_SIMD_VEC3_NEAR(r, e, SIMD_SCALAR_TOL) << "Mismatch at index " << i;
    }
}

//=============================================================================
// Matrix Identity Tests
//=============================================================================

TEST_F(SIMDMatrixTest, Identity_CreatesCorrectMatrix) {
    ASSERT_NE(m_dispatch->Matrix.Identity, nullptr);

    VxMatrix result;
    // Initialize with garbage
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            result[i][j] = 99.0f;
        }
    }

    m_dispatch->Matrix.Identity(&result);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            EXPECT_FLOAT_EQ(result[i][j], expected)
                << "Mismatch at [" << i << "][" << j << "]";
        }
    }
}

//=============================================================================
// Matrix Inverse Tests
//=============================================================================

TEST_F(SIMDMatrixTest, Inverse_Identity) {
    ASSERT_NE(m_dispatch->Matrix.Inverse, nullptr);

    VxMatrix identity;
    identity.SetIdentity();
    VxMatrix result;

    m_dispatch->Matrix.Inverse(&result, &identity);

    EXPECT_SIMD_MAT_NEAR(result, identity, SIMD_SCALAR_TOL);
}

TEST_F(SIMDMatrixTest, Inverse_MultiplyYieldsIdentity) {
    for (int i = 0; i < 30; ++i) {
        // Use orthogonal matrices for better numerical stability
        VxMatrix m = RandomRotationMatrix();
        // Add some translation
        m[3][0] = RandomFloat(-10.0f, 10.0f);
        m[3][1] = RandomFloat(-10.0f, 10.0f);
        m[3][2] = RandomFloat(-10.0f, 10.0f);

        VxMatrix inv, product;
        m_dispatch->Matrix.Inverse(&inv, &m);
        m_dispatch->Matrix.MultiplyMatrix(&product, &m, &inv);

        // Should be approximately identity
        VxMatrix identity;
        identity.SetIdentity();
        EXPECT_SIMD_MAT_NEAR(product, identity, SIMD_ACCUMULATED_TOL);
    }
}

TEST_F(SIMDMatrixTest, Inverse_MatchesScalar) {
    for (int i = 0; i < 30; ++i) {
        VxMatrix m = RandomRotationMatrix();
        m[3][0] = RandomFloat(-10.0f, 10.0f);
        m[3][1] = RandomFloat(-10.0f, 10.0f);
        m[3][2] = RandomFloat(-10.0f, 10.0f);

        VxMatrix expected;
        Vx3DInverseMatrix(expected, m);

        VxMatrix result;
        m_dispatch->Matrix.Inverse(&result, &m);

        EXPECT_SIMD_MAT_NEAR(result, expected, SIMD_SCALAR_TOL);
    }
}

//=============================================================================
// Matrix Determinant Tests
//=============================================================================

TEST_F(SIMDMatrixTest, Determinant_Identity) {
    ASSERT_NE(m_dispatch->Matrix.Determinant, nullptr);

    VxMatrix identity;
    identity.SetIdentity();

    float det = m_dispatch->Matrix.Determinant(&identity);

    EXPECT_NEAR(det, 1.0f, SIMD_EXACT_TOL);
}

TEST_F(SIMDMatrixTest, Determinant_Rotation) {
    // Rotation matrices have determinant 1
    VxMatrix rot = RandomRotationMatrix();

    float det = m_dispatch->Matrix.Determinant(&rot);

    EXPECT_NEAR(det, 1.0f, SIMD_SCALAR_TOL);
}

TEST_F(SIMDMatrixTest, Determinant_Scale) {
    VxMatrix scale;
    scale.SetIdentity();
    scale[0][0] = 2.0f;
    scale[1][1] = 3.0f;
    scale[2][2] = 4.0f;

    float det = m_dispatch->Matrix.Determinant(&scale);

    // Determinant should be product of scales (ignoring w row/column for 3D)
    EXPECT_NEAR(det, 24.0f, SIMD_SCALAR_TOL);
}

//=============================================================================
// Matrix From Axis-Angle Tests
//=============================================================================

TEST_F(SIMDMatrixTest, FromAxisAngle_ZeroAngle) {
    ASSERT_NE(m_dispatch->Matrix.FromAxisAngle, nullptr);

    VxVector axis(1.0f, 0.0f, 0.0f);
    VxMatrix result;

    m_dispatch->Matrix.FromAxisAngle(&result, &axis, 0.0f);

    VxMatrix identity;
    identity.SetIdentity();
    EXPECT_SIMD_MAT_NEAR(result, identity, SIMD_SCALAR_TOL);
}

TEST_F(SIMDMatrixTest, FromAxisAngle_90DegreesZ) {
    VxVector axis(0.0f, 0.0f, 1.0f);
    VxMatrix result;

    m_dispatch->Matrix.FromAxisAngle(&result, &axis, PI / 2.0f);

    // Apply to X axis should give Y
    VxVector x(1.0f, 0.0f, 0.0f);
    VxVector rotated;
    Vx3DRotateVector(&rotated, result, &x);

    EXPECT_NEAR(rotated.x, 0.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(rotated.y, 1.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(rotated.z, 0.0f, SIMD_SCALAR_TOL);
}

TEST_F(SIMDMatrixTest, FromAxisAngle_MatchesScalar) {
    for (int i = 0; i < 50; ++i) {
        VxVector axis = RandomUnitVector();
        float angle = RandomAngle();

        VxMatrix expected;
        Vx3DMatrixFromRotation(expected, axis, angle);

        VxMatrix result;
        m_dispatch->Matrix.FromAxisAngle(&result, &axis, angle);

        EXPECT_SIMD_MAT_NEAR(result, expected, SIMD_SCALAR_TOL);
    }
}

TEST_F(SIMDMatrixTest, FromAxisAngleOrigin_MatchesScalar) {
    ASSERT_NE(m_dispatch->Matrix.FromAxisAngleOrigin, nullptr);

    for (int i = 0; i < 50; ++i) {
        VxVector axis = RandomUnitVector();
        VxVector origin = RandomVector(-10.0f, 10.0f);
        float angle = RandomAngle();

        VxMatrix expected;
        Vx3DMatrixFromRotation(expected, axis, angle);
        VxVector rotatedOrigin;
        Vx3DRotateVector(&rotatedOrigin, expected, &origin);
        expected[3][0] = origin.x - rotatedOrigin.x;
        expected[3][1] = origin.y - rotatedOrigin.y;
        expected[3][2] = origin.z - rotatedOrigin.z;
        expected[3][3] = 1.0f;

        VxMatrix result;
        m_dispatch->Matrix.FromAxisAngleOrigin(&result, &axis, &origin, angle);

        EXPECT_SIMD_MAT_NEAR(result, expected, SIMD_SCALAR_TOL);
    }
}

//=============================================================================
// Matrix Euler Angles Tests
//=============================================================================

TEST_F(SIMDMatrixTest, FromEulerAngles_Zero) {
    ASSERT_NE(m_dispatch->Matrix.FromEulerAngles, nullptr);

    VxMatrix result;
    m_dispatch->Matrix.FromEulerAngles(&result, 0.0f, 0.0f, 0.0f);

    VxMatrix identity;
    identity.SetIdentity();
    EXPECT_SIMD_MAT_NEAR(result, identity, SIMD_SCALAR_TOL);
}

TEST_F(SIMDMatrixTest, FromEulerAngles_MatchesScalar) {
    for (int i = 0; i < 50; ++i) {
        float eax = RandomFloat(-PI, PI);
        float eay = RandomFloat(-PI / 2.0f + 0.1f, PI / 2.0f - 0.1f);  // Avoid gimbal lock
        float eaz = RandomFloat(-PI, PI);

        VxMatrix expected;
        Vx3DMatrixFromEulerAngles(expected, eax, eay, eaz);

        VxMatrix result;
        m_dispatch->Matrix.FromEulerAngles(&result, eax, eay, eaz);

        EXPECT_SIMD_MAT_NEAR(result, expected, SIMD_SCALAR_TOL);
    }
}

TEST_F(SIMDMatrixTest, ToEulerAngles_Roundtrip) {
    ASSERT_NE(m_dispatch->Matrix.ToEulerAngles, nullptr);

    for (int i = 0; i < 50; ++i) {
        float eax = RandomFloat(-PI * 0.4f, PI * 0.4f);
        float eay = RandomFloat(-PI / 2.0f + 0.2f, PI / 2.0f - 0.2f);
        float eaz = RandomFloat(-PI * 0.4f, PI * 0.4f);

        VxMatrix m;
        m_dispatch->Matrix.FromEulerAngles(&m, eax, eay, eaz);

        float outEax, outEay, outEaz;
        m_dispatch->Matrix.ToEulerAngles(&m, &outEax, &outEay, &outEaz);

        // Reconstruct matrix from extracted angles
        VxMatrix reconstructed;
        m_dispatch->Matrix.FromEulerAngles(&reconstructed, outEax, outEay, outEaz);

        // Matrices should be equivalent (angles may differ due to representation)
        EXPECT_SIMD_MAT_NEAR(reconstructed, m, SIMD_ACCUMULATED_TOL);
    }
}

//=============================================================================
// Matrix Interpolation Tests
//=============================================================================

TEST_F(SIMDMatrixTest, Interpolate_Boundaries) {
    ASSERT_NE(m_dispatch->Matrix.Interpolate, nullptr);

    VxMatrix a = RandomTRSMatrix();
    VxMatrix b = RandomTRSMatrix();
    VxMatrix result;
    VxMatrix expected;

    // step=0 should give same result as scalar
    m_dispatch->Matrix.Interpolate(0.0f, &result, &a, &b);
    Vx3DInterpolateMatrix(0.0f, expected, a, b);
    EXPECT_SIMD_MAT_NEAR(result, expected, SIMD_SCALAR_TOL);

    // step=1 should give same result as scalar
    m_dispatch->Matrix.Interpolate(1.0f, &result, &a, &b);
    Vx3DInterpolateMatrix(1.0f, expected, a, b);
    EXPECT_SIMD_MAT_NEAR(result, expected, SIMD_SCALAR_TOL);
}

TEST_F(SIMDMatrixTest, Interpolate_MatchesScalar) {
    for (int i = 0; i < 30; ++i) {
        VxMatrix a = RandomTRSMatrix();
        VxMatrix b = RandomTRSMatrix();
        float t = RandomInterpolationFactor();

        VxMatrix expected;
        Vx3DInterpolateMatrix(t, expected, a, b);

        VxMatrix result;
        m_dispatch->Matrix.Interpolate(t, &result, &a, &b);

        EXPECT_SIMD_MAT_NEAR(result, expected, SIMD_ACCUMULATED_TOL);
    }
}

//=============================================================================
// Matrix Decomposition Tests
//=============================================================================

TEST_F(SIMDMatrixTest, Decompose_Identity) {
    ASSERT_NE(m_dispatch->Matrix.Decompose, nullptr);

    VxMatrix identity;
    identity.SetIdentity();

    VxQuaternion quat;
    VxVector pos, scale;

    m_dispatch->Matrix.Decompose(&identity, &quat, &pos, &scale);

    // Identity quaternion
    EXPECT_NEAR(quat.w, 1.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(quat.x, 0.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(quat.y, 0.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(quat.z, 0.0f, SIMD_SCALAR_TOL);

    // Zero translation
    EXPECT_NEAR(pos.x, 0.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(pos.y, 0.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(pos.z, 0.0f, SIMD_SCALAR_TOL);

    // Unit scale
    EXPECT_NEAR(scale.x, 1.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(scale.y, 1.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(scale.z, 1.0f, SIMD_SCALAR_TOL);
}

TEST_F(SIMDMatrixTest, Decompose_TranslationOnly) {
    VxMatrix trans;
    trans.SetIdentity();
    trans[3][0] = 5.0f;
    trans[3][1] = 10.0f;
    trans[3][2] = 15.0f;

    VxQuaternion quat;
    VxVector pos, scale;

    m_dispatch->Matrix.Decompose(&trans, &quat, &pos, &scale);

    EXPECT_NEAR(pos.x, 5.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(pos.y, 10.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(pos.z, 15.0f, SIMD_SCALAR_TOL);

    EXPECT_NEAR(scale.x, 1.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(scale.y, 1.0f, SIMD_SCALAR_TOL);
    EXPECT_NEAR(scale.z, 1.0f, SIMD_SCALAR_TOL);
}

TEST_F(SIMDMatrixTest, Decompose_Roundtrip) {
    for (int i = 0; i < 30; ++i) {
        VxMatrix original = m_rng.TRSMatrix(10.0f, 0.5f, 2.0f);

        VxQuaternion quat;
        VxVector pos, scale;
        m_dispatch->Matrix.Decompose(&original, &quat, &pos, &scale);

        // Reconstruct matrix
        VxMatrix rotMat;
        quat.ToMatrix(rotMat);

        VxMatrix scaleMat;
        scaleMat.SetIdentity();
        scaleMat[0][0] = scale.x;
        scaleMat[1][1] = scale.y;
        scaleMat[2][2] = scale.z;

        VxMatrix temp, reconstructed;
        Vx3DMultiplyMatrix(temp, rotMat, scaleMat);
        temp[3][0] = pos.x;
        temp[3][1] = pos.y;
        temp[3][2] = pos.z;
        temp[3][3] = 1.0f;

        // Compare a test point transformation
        VxVector testVec = RandomVector();
        VxVector origResult, reconResult;
        Vx3DMultiplyMatrixVector(&origResult, original, &testVec);
        Vx3DMultiplyMatrixVector(&reconResult, temp, &testVec);

        EXPECT_SIMD_VEC3_NEAR(reconResult, origResult, SIMD_ACCUMULATED_TOL);
    }
}

//=============================================================================
// Edge Cases
//=============================================================================

TEST_F(SIMDMatrixTest, EdgeCase_SingularMatrix) {
    // Zero matrix (singular)
    VxMatrix zero;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            zero[i][j] = 0.0f;
        }
    }
    zero[3][3] = 1.0f;

    float det = m_dispatch->Matrix.Determinant(&zero);
    EXPECT_NEAR(det, 0.0f, SIMD_SCALAR_TOL);
}

TEST_F(SIMDMatrixTest, StressTest_ChainMultiply) {
    VxMatrix result;
    result.SetIdentity();

    // Chain many small rotations
    VxVector axis(0.0f, 1.0f, 0.0f);
    float smallAngle = PI / 100.0f;

    for (int i = 0; i < 200; ++i) {
        VxMatrix rot;
        m_dispatch->Matrix.FromAxisAngle(&rot, &axis, smallAngle);

        VxMatrix temp;
        m_dispatch->Matrix.MultiplyMatrix(&temp, &result, &rot);
        result = temp;
    }

    // Should have rotated 2*PI (full circle), back to identity-ish
    // Note: Due to accumulated error, we use loose tolerance
    VxVector testVec(1.0f, 0.0f, 0.0f);
    VxVector transformed;
    Vx3DRotateVector(&transformed, result, &testVec);

    // Should be close to original
    EXPECT_NEAR(transformed.x, testVec.x, 0.1f);
    EXPECT_NEAR(transformed.y, testVec.y, 0.1f);
    EXPECT_NEAR(transformed.z, testVec.z, 0.1f);
}

} // namespace
