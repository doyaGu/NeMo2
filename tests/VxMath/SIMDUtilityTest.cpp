/**
 * @file SIMDUtilityTest.cpp
 * @brief Tests for SIMD utility functions (load/store, dot product, cross product).
 *
 * Tests the inline helper functions defined in VxSIMD.h for SSE operations:
 * - VxSIMDLoadFloat3/VxSIMDStoreFloat3
 * - VxSIMDLoadFloat4/VxSIMDStoreFloat4
 * - VxSIMDLoadFloat4Aligned/VxSIMDStoreFloat4Aligned
 * - VxSIMDDotProduct3/VxSIMDDotProduct4
 * - VxSIMDCrossProduct3
 * - VxSIMDReciprocalSqrt/VxSIMDReciprocalSqrtAccurate
 * - VxSIMDNormalize3/VxSIMDNormalize4
 * - VxSIMDMatrixMultiplyVector3/VxSIMDMatrixRotateVector3
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>

#include "VxSIMD.h"
#include "SIMDTestCommon.h"

#if defined(VX_SIMD_SSE)

namespace {

using namespace SIMDTest;

//=============================================================================
// Load/Store Float3 Tests
//=============================================================================

TEST(SIMDUtility, LoadFloat3_StoreFloat3_Roundtrip) {
    const float input[3] = {1.0f, 2.0f, 3.0f};
    float output[3] = {0.0f, 0.0f, 0.0f};

    __m128 vec = VxSIMDLoadFloat3(input);
    VxSIMDStoreFloat3(output, vec);

    EXPECT_FLOAT_EQ(output[0], input[0]);
    EXPECT_FLOAT_EQ(output[1], input[1]);
    EXPECT_FLOAT_EQ(output[2], input[2]);
}

TEST(SIMDUtility, LoadFloat3_VariousValues) {
    const float testCases[][3] = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 1.0f},
        {-1.0f, -1.0f, -1.0f},
        {1e-6f, 1e-6f, 1e-6f},
        {1e6f, 1e6f, 1e6f},
        {-123.456f, 789.012f, -345.678f},
    };

    for (const auto& tc : testCases) {
        float output[3];
        __m128 vec = VxSIMDLoadFloat3(tc);
        VxSIMDStoreFloat3(output, vec);

        EXPECT_FLOAT_EQ(output[0], tc[0]) << "X mismatch";
        EXPECT_FLOAT_EQ(output[1], tc[1]) << "Y mismatch";
        EXPECT_FLOAT_EQ(output[2], tc[2]) << "Z mismatch";
    }
}

//=============================================================================
// Load/Store Float4 Tests
//=============================================================================

TEST(SIMDUtility, LoadFloat4_StoreFloat4_Roundtrip) {
    const float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    __m128 vec = VxSIMDLoadFloat4(input);
    VxSIMDStoreFloat4(output, vec);

    EXPECT_FLOAT_EQ(output[0], input[0]);
    EXPECT_FLOAT_EQ(output[1], input[1]);
    EXPECT_FLOAT_EQ(output[2], input[2]);
    EXPECT_FLOAT_EQ(output[3], input[3]);
}

TEST(SIMDUtility, LoadFloat4Aligned_StoreFloat4Aligned_Roundtrip) {
    alignas(16) float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    alignas(16) float output[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    __m128 vec = VxSIMDLoadFloat4Aligned(input);
    VxSIMDStoreFloat4Aligned(output, vec);

    EXPECT_FLOAT_EQ(output[0], input[0]);
    EXPECT_FLOAT_EQ(output[1], input[1]);
    EXPECT_FLOAT_EQ(output[2], input[2]);
    EXPECT_FLOAT_EQ(output[3], input[3]);
}

//=============================================================================
// Dot Product Tests
//=============================================================================

TEST(SIMDUtility, DotProduct3_UnitVectors) {
    // X dot X = 1
    __m128 x = _mm_set_ps(0.0f, 0.0f, 0.0f, 1.0f);
    __m128 dotX = VxSIMDDotProduct3(x, x);
    float resultX;
    _mm_store_ss(&resultX, dotX);
    EXPECT_NEAR(resultX, 1.0f, 1e-6f);

    // X dot Y = 0
    __m128 y = _mm_set_ps(0.0f, 0.0f, 1.0f, 0.0f);
    __m128 dotXY = VxSIMDDotProduct3(x, y);
    float resultXY;
    _mm_store_ss(&resultXY, dotXY);
    EXPECT_NEAR(resultXY, 0.0f, 1e-6f);
}

TEST(SIMDUtility, DotProduct3_VariousVectors) {
    VxMathTest::RandomGenerator rng(42);

    for (int i = 0; i < 100; ++i) {
        VxVector a = rng.Vector(-10.0f, 10.0f);
        VxVector b = rng.Vector(-10.0f, 10.0f);

        // Scalar reference
        const float expected = a.x * b.x + a.y * b.y + a.z * b.z;

        // SIMD
        __m128 va = VxSIMDLoadFloat3(&a.x);
        __m128 vb = VxSIMDLoadFloat3(&b.x);
        __m128 dot = VxSIMDDotProduct3(va, vb);
        float result;
        _mm_store_ss(&result, dot);

        EXPECT_TRUE(NearEqual(result, expected, 1e-5f))
            << "DotProduct3 mismatch: expected=" << expected << ", got=" << result;
    }
}

TEST(SIMDUtility, DotProduct4_UnitVectors) {
    __m128 x = _mm_set_ps(0.0f, 0.0f, 0.0f, 1.0f);
    __m128 dotX = VxSIMDDotProduct4(x, x);
    float resultX;
    _mm_store_ss(&resultX, dotX);
    EXPECT_NEAR(resultX, 1.0f, 1e-6f);

    __m128 ones = _mm_set1_ps(1.0f);
    __m128 dotOnes = VxSIMDDotProduct4(ones, ones);
    float resultOnes;
    _mm_store_ss(&resultOnes, dotOnes);
    EXPECT_NEAR(resultOnes, 4.0f, 1e-6f);
}

TEST(SIMDUtility, DotProduct4_VariousVectors) {
    VxMathTest::RandomGenerator rng(43);

    for (int i = 0; i < 100; ++i) {
        VxVector4 a = rng.Vector4(-10.0f, 10.0f);
        VxVector4 b = rng.Vector4(-10.0f, 10.0f);

        // Scalar reference
        const float expected = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;

        // SIMD
        __m128 va = VxSIMDLoadFloat4(&a.x);
        __m128 vb = VxSIMDLoadFloat4(&b.x);
        __m128 dot = VxSIMDDotProduct4(va, vb);
        float result;
        _mm_store_ss(&result, dot);

        EXPECT_TRUE(NearEqual(result, expected, 1e-5f))
            << "DotProduct4 mismatch: expected=" << expected << ", got=" << result;
    }
}

//=============================================================================
// Cross Product Tests
//=============================================================================

TEST(SIMDUtility, CrossProduct3_UnitVectors) {
    // X × Y = Z
    __m128 x = _mm_set_ps(0.0f, 0.0f, 0.0f, 1.0f);
    __m128 y = _mm_set_ps(0.0f, 0.0f, 1.0f, 0.0f);
    __m128 crossXY = VxSIMDCrossProduct3(x, y);

    float result[4];
    _mm_storeu_ps(result, crossXY);
    EXPECT_NEAR(result[0], 0.0f, 1e-6f) << "X × Y should have X=0";
    EXPECT_NEAR(result[1], 0.0f, 1e-6f) << "X × Y should have Y=0";
    EXPECT_NEAR(result[2], 1.0f, 1e-6f) << "X × Y should have Z=1";

    // Y × X = -Z
    __m128 crossYX = VxSIMDCrossProduct3(y, x);
    _mm_storeu_ps(result, crossYX);
    EXPECT_NEAR(result[0], 0.0f, 1e-6f) << "Y × X should have X=0";
    EXPECT_NEAR(result[1], 0.0f, 1e-6f) << "Y × X should have Y=0";
    EXPECT_NEAR(result[2], -1.0f, 1e-6f) << "Y × X should have Z=-1";

    // Y × Z = X
    __m128 z = _mm_set_ps(0.0f, 1.0f, 0.0f, 0.0f);
    __m128 crossYZ = VxSIMDCrossProduct3(y, z);
    _mm_storeu_ps(result, crossYZ);
    EXPECT_NEAR(result[0], 1.0f, 1e-6f) << "Y × Z should have X=1";
    EXPECT_NEAR(result[1], 0.0f, 1e-6f) << "Y × Z should have Y=0";
    EXPECT_NEAR(result[2], 0.0f, 1e-6f) << "Y × Z should have Z=0";
}

TEST(SIMDUtility, CrossProduct3_VariousVectors) {
    VxMathTest::RandomGenerator rng(44);

    for (int i = 0; i < 100; ++i) {
        VxVector a = rng.Vector(-10.0f, 10.0f);
        VxVector b = rng.Vector(-10.0f, 10.0f);

        // Scalar reference
        VxVector expected;
        Scalar::CrossProduct(expected, a, b);

        // SIMD
        __m128 va = VxSIMDLoadFloat3(&a.x);
        __m128 vb = VxSIMDLoadFloat3(&b.x);
        __m128 cross = VxSIMDCrossProduct3(va, vb);
        VxVector result;
        VxSIMDStoreFloat3(&result.x, cross);

        EXPECT_TRUE(VectorNear(result, expected, 1e-5f))
            << "CrossProduct3 mismatch";
    }
}

TEST(SIMDUtility, CrossProduct3_Perpendicular) {
    // Result should be perpendicular to both inputs
    VxMathTest::RandomGenerator rng(45);

    for (int i = 0; i < 50; ++i) {
        VxVector a = rng.UnitVector();
        VxVector b = rng.UnitVector();

        __m128 va = VxSIMDLoadFloat3(&a.x);
        __m128 vb = VxSIMDLoadFloat3(&b.x);
        __m128 cross = VxSIMDCrossProduct3(va, vb);

        // cross · a should be ~0
        __m128 dotA = VxSIMDDotProduct3(cross, va);
        float resultA;
        _mm_store_ss(&resultA, dotA);

        // cross · b should be ~0
        __m128 dotB = VxSIMDDotProduct3(cross, vb);
        float resultB;
        _mm_store_ss(&resultB, dotB);

        EXPECT_NEAR(resultA, 0.0f, 1e-5f) << "Cross product should be perpendicular to A";
        EXPECT_NEAR(resultB, 0.0f, 1e-5f) << "Cross product should be perpendicular to B";
    }
}

//=============================================================================
// Reciprocal Square Root Tests
//=============================================================================

TEST(SIMDUtility, ReciprocalSqrt_ApproximateAccuracy) {
    // rsqrt is fast but approximate (12-bit precision)
    const float testValues[] = {0.25f, 1.0f, 4.0f, 16.0f, 100.0f};

    for (float val : testValues) {
        __m128 v = _mm_set1_ps(val);
        __m128 rsqrt = VxSIMDReciprocalSqrt(v);
        float result;
        _mm_store_ss(&result, rsqrt);

        const float expected = 1.0f / std::sqrt(val);

        // rsqrt is only ~12 bits accurate
        EXPECT_NEAR(result, expected, expected * 0.01f)
            << "ReciprocalSqrt for " << val;
    }
}

TEST(SIMDUtility, ReciprocalSqrtAccurate_HighPrecision) {
    const float testValues[] = {0.25f, 1.0f, 4.0f, 16.0f, 100.0f, 1000.0f};

    for (float val : testValues) {
        __m128 v = _mm_set1_ps(val);
        __m128 rsqrt = VxSIMDReciprocalSqrtAccurate(v);
        float result;
        _mm_store_ss(&result, rsqrt);

        const float expected = 1.0f / std::sqrt(val);

        // Newton-Raphson refinement should give ~24 bits
        EXPECT_NEAR(result, expected, expected * 1e-5f)
            << "ReciprocalSqrtAccurate for " << val;
    }
}

//=============================================================================
// Normalize Tests
//=============================================================================

TEST(SIMDUtility, Normalize3_UnitVectors) {
    // Already normalized vectors should remain unchanged
    __m128 x = _mm_set_ps(0.0f, 0.0f, 0.0f, 1.0f);
    __m128 normX = VxSIMDNormalize3(x);

    float result[4];
    _mm_storeu_ps(result, normX);
    EXPECT_NEAR(result[0], 1.0f, 1e-5f);
    EXPECT_NEAR(result[1], 0.0f, 1e-5f);
    EXPECT_NEAR(result[2], 0.0f, 1e-5f);
}

TEST(SIMDUtility, Normalize3_VariousVectors) {
    VxMathTest::RandomGenerator rng(46);

    for (int i = 0; i < 100; ++i) {
        VxVector v = rng.Vector(-10.0f, 10.0f);
        if (v.SquareMagnitude() < EPSILON) continue;

        __m128 vec = VxSIMDLoadFloat3(&v.x);
        __m128 norm = VxSIMDNormalize3(vec);

        VxVector result;
        VxSIMDStoreFloat3(&result.x, norm);

        // Check magnitude is ~1
        const float mag = std::sqrt(result.x * result.x + result.y * result.y + result.z * result.z);
        EXPECT_NEAR(mag, 1.0f, 1e-5f) << "Normalized vector should have magnitude 1";

        // Check direction is preserved
        VxVector expected = v;
        expected.Normalize();
        EXPECT_TRUE(VectorNear(result, expected, 1e-5f)) << "Normalize3 direction mismatch";
    }
}

TEST(SIMDUtility, Normalize3_ZeroVector) {
    // Zero vector normalization should not crash and return zero
    __m128 zero = _mm_setzero_ps();
    __m128 norm = VxSIMDNormalize3(zero);

    float result[4];
    _mm_storeu_ps(result, norm);

    // Should return zero vector (not NaN)
    EXPECT_FALSE(std::isnan(result[0]));
    EXPECT_FALSE(std::isnan(result[1]));
    EXPECT_FALSE(std::isnan(result[2]));
}

TEST(SIMDUtility, Normalize4_VariousVectors) {
    VxMathTest::RandomGenerator rng(47);

    for (int i = 0; i < 100; ++i) {
        VxVector4 v = rng.Vector4(-10.0f, 10.0f);
        const float lenSq = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        if (lenSq < EPSILON) continue;

        __m128 vec = VxSIMDLoadFloat4(&v.x);
        __m128 norm = VxSIMDNormalize4(vec);

        VxVector4 result;
        VxSIMDStoreFloat4(&result.x, norm);

        // Check magnitude is ~1
        const float mag = std::sqrt(result.x * result.x + result.y * result.y +
                                   result.z * result.z + result.w * result.w);
        EXPECT_NEAR(mag, 1.0f, 1e-5f) << "Normalized vector4 should have magnitude 1";
    }
}

//=============================================================================
// Matrix-Vector Multiply Tests
//=============================================================================

TEST(SIMDUtility, MatrixMultiplyVector3_Identity) {
    VxMatrix identity;
    identity.SetIdentity();

    VxVector v(1.0f, 2.0f, 3.0f);
    __m128 vec = VxSIMDLoadFloat3(&v.x);
    __m128 result = VxSIMDMatrixMultiplyVector3((const float*)&identity[0][0], vec);

    VxVector out;
    VxSIMDStoreFloat3(&out.x, result);

    EXPECT_NEAR(out.x, v.x, 1e-6f);
    EXPECT_NEAR(out.y, v.y, 1e-6f);
    EXPECT_NEAR(out.z, v.z, 1e-6f);
}

TEST(SIMDUtility, MatrixMultiplyVector3_Translation) {
    VxMatrix trans;
    trans.SetIdentity();
    trans[3][0] = 10.0f;
    trans[3][1] = 20.0f;
    trans[3][2] = 30.0f;

    VxVector v(1.0f, 2.0f, 3.0f);
    __m128 vec = VxSIMDLoadFloat3(&v.x);
    __m128 result = VxSIMDMatrixMultiplyVector3((const float*)&trans[0][0], vec);

    VxVector out;
    VxSIMDStoreFloat3(&out.x, result);

    EXPECT_NEAR(out.x, 11.0f, 1e-6f);
    EXPECT_NEAR(out.y, 22.0f, 1e-6f);
    EXPECT_NEAR(out.z, 33.0f, 1e-6f);
}

TEST(SIMDUtility, MatrixRotateVector3_Identity) {
    VxMatrix identity;
    identity.SetIdentity();

    VxVector v(1.0f, 2.0f, 3.0f);
    __m128 vec = VxSIMDLoadFloat3(&v.x);
    __m128 result = VxSIMDMatrixRotateVector3((const float*)&identity[0][0], vec);

    VxVector out;
    VxSIMDStoreFloat3(&out.x, result);

    EXPECT_NEAR(out.x, v.x, 1e-6f);
    EXPECT_NEAR(out.y, v.y, 1e-6f);
    EXPECT_NEAR(out.z, v.z, 1e-6f);
}

TEST(SIMDUtility, MatrixRotateVector3_IgnoresTranslation) {
    VxMatrix trans;
    trans.SetIdentity();
    trans[3][0] = 100.0f;
    trans[3][1] = 200.0f;
    trans[3][2] = 300.0f;

    VxVector v(1.0f, 2.0f, 3.0f);
    __m128 vec = VxSIMDLoadFloat3(&v.x);
    __m128 result = VxSIMDMatrixRotateVector3((const float*)&trans[0][0], vec);

    VxVector out;
    VxSIMDStoreFloat3(&out.x, result);

    // Should NOT include translation
    EXPECT_NEAR(out.x, v.x, 1e-6f);
    EXPECT_NEAR(out.y, v.y, 1e-6f);
    EXPECT_NEAR(out.z, v.z, 1e-6f);
}

TEST(SIMDUtility, MatrixRotateVector3_RandomRotations) {
    VxMathTest::RandomGenerator rng(48);

    for (int i = 0; i < 50; ++i) {
        VxMatrix rot = rng.RotationMatrix();
        VxVector v = rng.Vector(-10.0f, 10.0f);

        // Scalar reference
        VxVector expected;
        Vx3DRotateVector(&expected, rot, &v);

        // SIMD
        __m128 vec = VxSIMDLoadFloat3(&v.x);
        __m128 result = VxSIMDMatrixRotateVector3((const float*)&rot[0][0], vec);
        VxVector out;
        VxSIMDStoreFloat3(&out.x, result);

        EXPECT_TRUE(VectorNear(out, expected, 1e-5f))
            << "MatrixRotateVector3 mismatch";
    }
}

//=============================================================================
// Edge Cases
//=============================================================================

TEST(SIMDUtility, EdgeCase_VerySmallValues) {
    const float tiny = 1e-38f;
    float input[3] = {tiny, tiny, tiny};
    float output[3];

    __m128 vec = VxSIMDLoadFloat3(input);
    VxSIMDStoreFloat3(output, vec);

    EXPECT_FLOAT_EQ(output[0], tiny);
    EXPECT_FLOAT_EQ(output[1], tiny);
    EXPECT_FLOAT_EQ(output[2], tiny);
}

TEST(SIMDUtility, EdgeCase_VeryLargeValues) {
    const float big = 1e38f;
    float input[3] = {big, -big, big};
    float output[3];

    __m128 vec = VxSIMDLoadFloat3(input);
    VxSIMDStoreFloat3(output, vec);

    EXPECT_FLOAT_EQ(output[0], big);
    EXPECT_FLOAT_EQ(output[1], -big);
    EXPECT_FLOAT_EQ(output[2], big);
}

TEST(SIMDUtility, EdgeCase_MixedSigns) {
    float input[4] = {1.0f, -2.0f, 3.0f, -4.0f};
    float output[4];

    __m128 vec = VxSIMDLoadFloat4(input);
    VxSIMDStoreFloat4(output, vec);

    EXPECT_FLOAT_EQ(output[0], 1.0f);
    EXPECT_FLOAT_EQ(output[1], -2.0f);
    EXPECT_FLOAT_EQ(output[2], 3.0f);
    EXPECT_FLOAT_EQ(output[3], -4.0f);
}

} // namespace

#endif // VX_SIMD_SSE
