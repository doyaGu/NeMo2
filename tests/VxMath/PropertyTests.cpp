/**
 * @file PropertyTests.cpp
 * @brief Property-based tests for VxMath library.
 *
 * This file contains property-based tests that validate mathematical invariants
 * across many random inputs. Unlike specific unit tests, property tests verify
 * that fundamental mathematical properties hold for arbitrary valid inputs.
 */

#include <gtest/gtest.h>
#include "VxMathTestHelpers.h"

using namespace VxMathTest;

//=============================================================================
// Test Fixture
//=============================================================================

class PropertyTest : public ::testing::Test {
protected:
    static constexpr int ITERATIONS = 100;
    static constexpr uint32_t SEED = 42;
};

//=============================================================================
// Vector Property Tests
//=============================================================================

TEST_F(PropertyTest, VectorNormalizationProducesUnitLength) {
    TestProperty<VxVector>(
        "NormalizationProducesUnitLength",
        VectorProperties::NormalizationProducesUnitLength,
        [](RandomGenerator& gen) { return gen.Vector(); },
        ITERATIONS, SEED
    );
}

TEST_F(PropertyTest, VectorMagnitudeIsNonNegative) {
    TestProperty<VxVector>(
        "MagnitudeIsNonNegative",
        VectorProperties::MagnitudeIsNonNegative,
        [](RandomGenerator& gen) { return gen.Vector(); },
        ITERATIONS, SEED
    );
}

TEST_F(PropertyTest, VectorDotProductIsCommutative) {
    TestProperty2<VxVector, VxVector>(
        "DotProductIsCommutative",
        VectorProperties::DotProductIsCommutative,
        [](RandomGenerator& gen) { return gen.Vector(); },
        [](RandomGenerator& gen) { return gen.Vector(); },
        ITERATIONS, SEED
    );
}

TEST_F(PropertyTest, VectorCrossProductIsAntiCommutative) {
    TestProperty2<VxVector, VxVector>(
        "CrossProductIsAntiCommutative",
        VectorProperties::CrossProductIsAntiCommutative,
        [](RandomGenerator& gen) { return gen.Vector(); },
        [](RandomGenerator& gen) { return gen.Vector(); },
        ITERATIONS, SEED
    );
}

TEST_F(PropertyTest, VectorNormalizedDotSelfIsOne) {
    TestProperty<VxVector>(
        "NormalizedVectorDotSelfIsOne",
        VectorProperties::NormalizedVectorDotSelfIsOne,
        [](RandomGenerator& gen) { return gen.Vector(); },
        ITERATIONS, SEED
    );
}

TEST_F(PropertyTest, VectorCrossProductIsOrthogonal) {
    TestProperty2<VxVector, VxVector>(
        "CrossProductIsOrthogonal",
        VectorProperties::CrossProductIsOrthogonal,
        [](RandomGenerator& gen) { return gen.Vector(); },
        [](RandomGenerator& gen) { return gen.Vector(); },
        ITERATIONS, SEED
    );
}

TEST_F(PropertyTest, VectorAdditionIsCommutative) {
    TestProperty2<VxVector, VxVector>(
        "AdditionIsCommutative",
        VectorProperties::AdditionIsCommutative,
        [](RandomGenerator& gen) { return gen.Vector(); },
        [](RandomGenerator& gen) { return gen.Vector(); },
        ITERATIONS, SEED
    );
}

//=============================================================================
// Matrix Property Tests
//=============================================================================

TEST_F(PropertyTest, MatrixIdentityMultiplicationIsIdentity) {
    TestProperty<VxMatrix>(
        "IdentityMultiplicationIsIdentity",
        MatrixProperties::IdentityMultiplicationIsIdentity,
        [](RandomGenerator& gen) { return gen.TRSMatrix(); },
        ITERATIONS, SEED
    );
}

TEST_F(PropertyTest, MatrixInverseOfInverseIsOriginal) {
    TestProperty<VxMatrix>(
        "InverseOfInverseIsOriginal",
        MatrixProperties::InverseOfInverseIsOriginal,
        [](RandomGenerator& gen) { return gen.TRSMatrix(); },
        ITERATIONS, SEED
    );
}

TEST_F(PropertyTest, MatrixTransposeOfTransposeIsOriginal) {
    TestProperty<VxMatrix>(
        "TransposeOfTransposeIsOriginal",
        MatrixProperties::TransposeOfTransposeIsOriginal,
        [](RandomGenerator& gen) { return gen.TRSMatrix(); },
        ITERATIONS, SEED
    );
}

TEST_F(PropertyTest, MatrixInverseMultiplicationIsIdentity) {
    TestProperty<VxMatrix>(
        "InverseMultiplicationIsIdentity",
        MatrixProperties::InverseMultiplicationIsIdentity,
        [](RandomGenerator& gen) { return gen.TRSMatrix(); },
        ITERATIONS, SEED
    );
}

TEST_F(PropertyTest, RotationMatrixHasUnitDeterminant) {
    TestProperty<VxMatrix>(
        "RotationMatrixHasUnitDeterminant",
        MatrixProperties::RotationMatrixHasUnitDeterminant,
        [](RandomGenerator& gen) { return gen.RotationMatrix(); },
        ITERATIONS, SEED
    );
}

TEST_F(PropertyTest, RotationPreservesMagnitude) {
    TestProperty2<VxMatrix, VxVector>(
        "RotationPreservesMagnitude",
        MatrixProperties::RotationPreservesMagnitude,
        [](RandomGenerator& gen) { return gen.RotationMatrix(); },
        [](RandomGenerator& gen) { return gen.Vector(); },
        ITERATIONS, SEED
    );
}

//=============================================================================
// Quaternion Property Tests
//=============================================================================

TEST_F(PropertyTest, QuaternionNormalizedHasUnitMagnitude) {
    TestProperty<VxQuaternion>(
        "NormalizedQuaternionHasUnitMagnitude",
        QuaternionProperties::NormalizedQuaternionHasUnitMagnitude,
        [](RandomGenerator& gen) { return gen.ArbitraryQuaternion(); },
        ITERATIONS, SEED
    );
}

TEST_F(PropertyTest, QuaternionAndNegativeRepresentSameRotation) {
    TestProperty<VxQuaternion>(
        "QuaternionAndNegativeRepresentSameRotation",
        QuaternionProperties::QuaternionAndNegativeRepresentSameRotation,
        [](RandomGenerator& gen) { return gen.UnitQuaternion(); },
        ITERATIONS, SEED
    );
}

TEST_F(PropertyTest, QuaternionSlerpEndpointsMatch) {
    TestProperty2<VxQuaternion, VxQuaternion>(
        "SlerpEndpointsMatch",
        QuaternionProperties::SlerpEndpointsMatch,
        [](RandomGenerator& gen) { return gen.UnitQuaternion(); },
        [](RandomGenerator& gen) { return gen.UnitQuaternion(); },
        ITERATIONS, SEED
    );
}

TEST_F(PropertyTest, QuaternionConjugateProductIsIdentity) {
    TestProperty<VxQuaternion>(
        "ConjugateProductIsIdentity",
        QuaternionProperties::ConjugateProductIsIdentity,
        [](RandomGenerator& gen) { return gen.UnitQuaternion(); },
        ITERATIONS, SEED
    );
}

TEST_F(PropertyTest, QuaternionMultiplicationIsAssociative) {
    RandomGenerator rng(SEED);

    for (int i = 0; i < ITERATIONS; ++i) {
        VxQuaternion a = rng.UnitQuaternion();
        VxQuaternion b = rng.UnitQuaternion();
        VxQuaternion c = rng.UnitQuaternion();

        ::testing::AssertionResult result = QuaternionProperties::MultiplicationIsAssociative(a, b, c);
        EXPECT_TRUE(result) << "Property 'MultiplicationIsAssociative' failed on iteration " << i
                            << " (seed=" << SEED << ")";
        if (!result) return;
    }
}

TEST_F(PropertyTest, QuaternionDivideUndoesMultiplyForUnitQuaternions) {
    RandomGenerator rng(SEED);

    for (int i = 0; i < ITERATIONS; ++i) {
        VxQuaternion a = rng.UnitQuaternion();
        VxQuaternion b = rng.UnitQuaternion();

        VxQuaternion product = Vx3DQuaternionMultiply(a, b);
        VxQuaternion recovered = Vx3DQuaternionDivide(product, b);

        EXPECT_TRUE(QuaternionNearBool(recovered, a, ACCUMULATION_TOL))
            << "(a*b)/b != a on iteration " << i << " (seed=" << SEED << ")\n"
            << "a=" << QuaternionToString(a) << "\n"
            << "b=" << QuaternionToString(b) << "\n"
            << "recovered=" << QuaternionToString(recovered);
    }
}

TEST_F(PropertyTest, QuaternionLnExpRoundTripForUnitQuaternions) {
    RandomGenerator rng(SEED);

    for (int i = 0; i < ITERATIONS; ++i) {
        VxQuaternion q = rng.UnitQuaternion();

        // Avoid the branch-cut/degenerate axis case: q ~= (-1, 0, 0, 0) where
        // the vector part is ~0 but w < 0, making ln(q) ill-defined.
        float vMag = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
        if (vMag < 1e-4f && q.w < 0.0f) {
            continue;
        }

        VxQuaternion lnq = Ln(q);
        VxQuaternion exp_lq = Exp(lnq);

        EXPECT_TRUE(QuaternionNearBool(exp_lq, q, ACCUMULATION_TOL))
            << "exp(ln(q)) != q on iteration " << i << " (seed=" << SEED << ")\n"
            << "q=" << QuaternionToString(q) << "\n"
            << "exp(ln(q))=" << QuaternionToString(exp_lq);
    }
}

TEST_F(PropertyTest, QuaternionLnDifMatchesLnOfDivide) {
    RandomGenerator rng(SEED);

    for (int i = 0; i < ITERATIONS; ++i) {
        VxQuaternion p = rng.UnitQuaternion();
        VxQuaternion q = rng.UnitQuaternion();

        VxQuaternion expected = Ln(Vx3DQuaternionDivide(q, p));
        VxQuaternion actual = LnDif(p, q);

        EXPECT_TRUE(QuaternionNearBool(actual, expected, ACCUMULATION_TOL))
            << "LnDif(p, q) != Ln(q/p) on iteration " << i << " (seed=" << SEED << ")\n"
            << "p=" << QuaternionToString(p) << "\n"
            << "q=" << QuaternionToString(q) << "\n"
            << "actual=" << QuaternionToString(actual) << "\n"
            << "expected=" << QuaternionToString(expected);
    }
}

TEST_F(PropertyTest, QuaternionMatrixRoundTripPreservesRotation) {
    RandomGenerator rng(SEED);

    for (int i = 0; i < ITERATIONS; ++i) {
        VxQuaternion original = rng.UnitQuaternion();
        VxMatrix m;
        original.ToMatrix(m);

        VxQuaternion reconstructed = Vx3DQuaternionFromMatrix(m);
        EXPECT_TRUE(QuaternionNearBool(reconstructed, original, ACCUMULATION_TOL))
            << "FromMatrix(ToMatrix(q)) != q on iteration " << i << " (seed=" << SEED << ")\n"
            << "original=" << QuaternionToString(original) << "\n"
            << "reconstructed=" << QuaternionToString(reconstructed);
    }
}

TEST_F(PropertyTest, QuaternionSlerpProducesUnitQuaternion) {
    RandomGenerator rng(SEED);

    for (int i = 0; i < ITERATIONS; ++i) {
        VxQuaternion a = rng.UnitQuaternion();
        VxQuaternion b = rng.UnitQuaternion();
        float t = rng.InterpolationFactor();

        VxQuaternion s = Slerp(t, a, b);
        float len = std::sqrt(s.x * s.x + s.y * s.y + s.z * s.z + s.w * s.w);

        EXPECT_TRUE(ScaleRelativeNear(len, 1.0f, ACCUMULATION_TOL))
            << "|Slerp(t,a,b)| != 1 on iteration " << i << " (seed=" << SEED << ")\n"
            << "t=" << t << "\n"
            << "len=" << len;
    }
}

TEST_F(PropertyTest, QuaternionToMatrixHasUnitDeterminant) {
    RandomGenerator rng(SEED);

    for (int i = 0; i < ITERATIONS; ++i) {
        VxQuaternion q = rng.UnitQuaternion();
        VxMatrix m;
        q.ToMatrix(m);

        float det = Vx3DMatrixDeterminant(m);
        EXPECT_TRUE(ScaleRelativeNear(det, 1.0f, LOOSE_TOL))
            << "det(ToMatrix(q)) != 1 on iteration " << i << " (seed=" << SEED << ")\n"
            << "det=" << det;
    }
}

//=============================================================================
// Geometric Property Tests
//=============================================================================

TEST_F(PropertyTest, BboxContainsItsCenter) {
    TestProperty<VxBbox>(
        "BboxContainsItsCenter",
        GeometricProperties::BboxContainsItsCenter,
        [](RandomGenerator& gen) { return gen.Bbox(); },
        ITERATIONS, SEED
    );
}

TEST_F(PropertyTest, BboxMergingWithSelfIsIdempotent) {
    TestProperty<VxBbox>(
        "MergingBboxWithSelfIsIdempotent",
        GeometricProperties::MergingBboxWithSelfIsIdempotent,
        [](RandomGenerator& gen) { return gen.Bbox(); },
        ITERATIONS, SEED
    );
}

TEST_F(PropertyTest, SphereContainsItsCenter) {
    TestProperty<VxSphere>(
        "SphereContainsItsCenter",
        GeometricProperties::SphereContainsItsCenter,
        [](RandomGenerator& gen) { return gen.Sphere(); },
        ITERATIONS, SEED
    );
}

//=============================================================================
// Additional Derived Property Tests
//=============================================================================

/// Test that normalizing a unit vector is idempotent
TEST_F(PropertyTest, NormalizingUnitVectorIsIdempotent) {
    TestProperty<VxVector>(
        "NormalizingUnitVectorIsIdempotent",
        [](const VxVector& v) -> ::testing::AssertionResult {
            if (v.SquareMagnitude() < 1e-20f) {
                return ::testing::AssertionSuccess();
            }
            VxVector n = Normalize(v);
            VxVector nn = Normalize(n);
            if (!VectorNearBool(n, nn, SIMD_TOL)) {
                return ::testing::AssertionFailure()
                    << "Normalize(Normalize(v)) != Normalize(v)";
            }
            return ::testing::AssertionSuccess();
        },
        [](RandomGenerator& gen) { return gen.Vector(); },
        ITERATIONS, SEED
    );
}

/// Test that matrix multiplication is associative
TEST_F(PropertyTest, MatrixMultiplicationIsAssociative) {
    RandomGenerator rng(SEED);

    for (int i = 0; i < ITERATIONS; ++i) {
        VxMatrix a = rng.TRSMatrix();
        VxMatrix b = rng.TRSMatrix();
        VxMatrix c = rng.TRSMatrix();

        VxMatrix ab, ab_c;
        Vx3DMultiplyMatrix(ab, a, b);
        Vx3DMultiplyMatrix(ab_c, ab, c);

        VxMatrix bc, a_bc;
        Vx3DMultiplyMatrix(bc, b, c);
        Vx3DMultiplyMatrix(a_bc, a, bc);

        // Use looser tolerance due to floating-point accumulation
        EXPECT_TRUE(MatrixNearBool(ab_c, a_bc, ACCUMULATION_TOL))
            << "(A*B)*C != A*(B*C) on iteration " << i;
    }
}

/// Test that lerp endpoints match
TEST_F(PropertyTest, VectorLerpEndpointsMatch) {
    RandomGenerator rng(SEED);

    for (int i = 0; i < ITERATIONS; ++i) {
        VxVector a = rng.Vector();
        VxVector b = rng.Vector();

        VxVector lerp0 = a + (b - a) * 0.0f;
        VxVector lerp1 = a + (b - a) * 1.0f;

        EXPECT_TRUE(VectorNearBool(lerp0, a, STRICT_TOL))
            << "Lerp(0) != a on iteration " << i;
        EXPECT_TRUE(VectorNearBool(lerp1, b, STRICT_TOL))
            << "Lerp(1) != b on iteration " << i;
    }
}

/// Test that quaternion-matrix conversion round-trips
TEST_F(PropertyTest, QuaternionMatrixRoundTrip) {
    RandomGenerator rng(SEED);

    for (int i = 0; i < ITERATIONS; ++i) {
        VxQuaternion original = rng.UnitQuaternion();

        VxMatrix mat;
        original.ToMatrix(mat);

        VxQuaternion reconstructed;
        reconstructed.FromMatrix(mat);

        EXPECT_TRUE(QuaternionNearBool(original, reconstructed, ACCUMULATION_TOL))
            << "Quaternion->Matrix->Quaternion round trip failed on iteration " << i;
    }
}

/// Test that scaling a vector scales its magnitude
TEST_F(PropertyTest, VectorScalingScalesMagnitude) {
    RandomGenerator rng(SEED);

    for (int i = 0; i < ITERATIONS; ++i) {
        VxVector v = rng.Vector();
        float scale = rng.Scale(0.1f, 10.0f);

        float origMag = v.Magnitude();
        float scaledMag = (v * scale).Magnitude();
        float expectedMag = origMag * scale;

        EXPECT_TRUE(ScaleRelativeNear(scaledMag, expectedMag, SIMD_TOL))
            << "|v*s| = " << scaledMag << " but |v|*s = " << expectedMag
            << " on iteration " << i;
    }
}

/// Test that orthogonal matrix inverse equals transpose
TEST_F(PropertyTest, OrthogonalMatrixInverseEqualsTranspose) {
    RandomGenerator rng(SEED);

    for (int i = 0; i < ITERATIONS; ++i) {
        VxMatrix rot = rng.RotationMatrix();

        VxMatrix inv, trans;
        Vx3DInverseMatrix(inv, rot);
        Vx3DTransposeMatrix(trans, rot);

        EXPECT_TRUE(MatrixNearBool(inv, trans, SIMD_TOL))
            << "R^-1 != R^T for rotation matrix on iteration " << i;
    }
}

/// Test that bbox merge is commutative
TEST_F(PropertyTest, BboxMergeIsCommutative) {
    RandomGenerator rng(SEED);

    for (int i = 0; i < ITERATIONS; ++i) {
        VxBbox a = rng.Bbox();
        VxBbox b = rng.Bbox();

        VxBbox ab = a;
        ab.Merge(b);

        VxBbox ba = b;
        ba.Merge(a);

        EXPECT_TRUE(BboxNearBool(ab, ba, STRICT_TOL))
            << "a.Merge(b) != b.Merge(a) on iteration " << i;
    }
}

/// Test that sphere-ray intersection is symmetric for point inside sphere
TEST_F(PropertyTest, SphereContainsPointsWithinRadius) {
    RandomGenerator rng(SEED);

    for (int i = 0; i < ITERATIONS; ++i) {
        VxSphere sphere = rng.Sphere();
        VxVector center = sphere.Center();
        float radius = sphere.Radius();

        // Generate a random point inside the sphere
        VxVector dir = rng.UnitVector();
        float dist = rng.Float(0.0f, radius * 0.99f);
        VxVector pointInside = center + dir * dist;

        // Check that distance from center is within radius
        float actualDist = (pointInside - center).Magnitude();
        EXPECT_LE(actualDist, radius)
            << "Point supposedly inside sphere is outside on iteration " << i;
    }
}
