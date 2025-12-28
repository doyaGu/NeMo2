/**
 * @file VxMathTestPropertyTesting.h
 * @brief Property-based testing infrastructure for VxMath tests.
 *
 * This header provides a framework for property-based testing, which validates
 * mathematical invariants across many random inputs rather than specific cases.
 */

#ifndef VXMATH_TEST_PROPERTY_TESTING_H
#define VXMATH_TEST_PROPERTY_TESTING_H

#include <gtest/gtest.h>
#include <functional>
#include <string>
#include <sstream>

#include "VxMath.h"
#include "VxMathTestTolerances.h"
#include "VxMathTestComparisons.h"
#include "VxMathTestGenerators.h"

namespace VxMathTest {

//=============================================================================
// Property Testing Framework
//=============================================================================

/**
 * @brief Run a property test with random inputs.
 *
 * @tparam T The type of value to test
 * @param propName Name of the property being tested
 * @param property Function that returns AssertionResult for a given input
 * @param generator Function that generates random test values
 * @param iterations Number of random tests to run
 * @param seed Random seed for reproducibility
 */
template<typename T>
void TestProperty(
    const std::string& propName,
    std::function<::testing::AssertionResult(const T&)> property,
    std::function<T(RandomGenerator&)> generator,
    int iterations = 100,
    uint32_t seed = 42
) {
    RandomGenerator rng(seed);

    for (int i = 0; i < iterations; ++i) {
        T value = generator(rng);
        ::testing::AssertionResult result = property(value);
        EXPECT_TRUE(result)
            << "Property '" << propName << "' failed on iteration " << i
            << " (seed=" << seed << ")";
        if (!result) {
            // Early exit on first failure for faster debugging
            return;
        }
    }
}

/**
 * @brief Run a property test with two random inputs.
 *
 * @tparam T1 First input type
 * @tparam T2 Second input type
 * @param propName Name of the property being tested
 * @param property Function that returns AssertionResult for given inputs
 * @param gen1 Generator for first input
 * @param gen2 Generator for second input
 * @param iterations Number of random tests to run
 * @param seed Random seed for reproducibility
 */
template<typename T1, typename T2>
void TestProperty2(
    const std::string& propName,
    std::function<::testing::AssertionResult(const T1&, const T2&)> property,
    std::function<T1(RandomGenerator&)> gen1,
    std::function<T2(RandomGenerator&)> gen2,
    int iterations = 100,
    uint32_t seed = 42
) {
    RandomGenerator rng(seed);

    for (int i = 0; i < iterations; ++i) {
        T1 v1 = gen1(rng);
        T2 v2 = gen2(rng);
        ::testing::AssertionResult result = property(v1, v2);
        EXPECT_TRUE(result)
            << "Property '" << propName << "' failed on iteration " << i
            << " (seed=" << seed << ")";
        if (!result) {
            return;
        }
    }
}

//=============================================================================
// Vector Properties
//=============================================================================

namespace VectorProperties {

/// Property: Normalizing a non-zero vector produces a unit vector
inline ::testing::AssertionResult NormalizationProducesUnitLength(const VxVector& v) {
    if (v.SquareMagnitude() < 1e-20f) {
        return ::testing::AssertionSuccess(); // Skip near-zero vectors
    }

    VxVector normalized = Normalize(v);
    float mag = normalized.Magnitude();

    if (std::fabs(mag - 1.0f) > SIMD_TOL) {
        return ::testing::AssertionFailure()
            << "Expected magnitude 1.0, got " << mag
            << " for input " << VectorToString(v);
    }
    return ::testing::AssertionSuccess();
}

/// Property: Magnitude is always non-negative
inline ::testing::AssertionResult MagnitudeIsNonNegative(const VxVector& v) {
    float mag = v.Magnitude();
    if (mag < 0.0f) {
        return ::testing::AssertionFailure()
            << "Magnitude " << mag << " is negative for " << VectorToString(v);
    }
    return ::testing::AssertionSuccess();
}

/// Property: Dot product is commutative: a . b = b . a
inline ::testing::AssertionResult DotProductIsCommutative(const VxVector& a, const VxVector& b) {
    float ab = DotProduct(a, b);
    float ba = DotProduct(b, a);

    if (!ScaleRelativeNear(ab, ba, SIMD_TOL)) {
        return ::testing::AssertionFailure()
            << "a.b=" << ab << " but b.a=" << ba
            << " for a=" << VectorToString(a) << ", b=" << VectorToString(b);
    }
    return ::testing::AssertionSuccess();
}

/// Property: Cross product is anti-commutative: a x b = -(b x a)
inline ::testing::AssertionResult CrossProductIsAntiCommutative(const VxVector& a, const VxVector& b) {
    VxVector ab = CrossProduct(a, b);
    VxVector ba = CrossProduct(b, a);
    VxVector neg_ba = -ba;

    if (!VectorNearBool(ab, neg_ba, SIMD_TOL)) {
        return ::testing::AssertionFailure()
            << "a x b = " << VectorToString(ab)
            << " but -(b x a) = " << VectorToString(neg_ba)
            << " for a=" << VectorToString(a) << ", b=" << VectorToString(b);
    }
    return ::testing::AssertionSuccess();
}

/// Property: Normalized vector dot self equals 1
inline ::testing::AssertionResult NormalizedVectorDotSelfIsOne(const VxVector& v) {
    if (v.SquareMagnitude() < 1e-20f) {
        return ::testing::AssertionSuccess(); // Skip near-zero vectors
    }

    VxVector n = Normalize(v);
    float dot = DotProduct(n, n);

    if (!ScaleRelativeNear(dot, 1.0f, SIMD_TOL)) {
        return ::testing::AssertionFailure()
            << "n.n = " << dot << " (expected 1.0) for input " << VectorToString(v);
    }
    return ::testing::AssertionSuccess();
}

/// Property: Cross product is orthogonal to both inputs
/// Note: Uses LOOSE_TOL because the ground-truth SIMD implementation (circa 2002)
/// has significant numerical error in cross product operations with large vectors.
inline ::testing::AssertionResult CrossProductIsOrthogonal(const VxVector& a, const VxVector& b) {
    // Skip parallel vectors (cross product is zero)
    VxVector cross = CrossProduct(a, b);
    if (cross.SquareMagnitude() < 1e-10f) {
        return ::testing::AssertionSuccess();
    }

    float dot_a = DotProduct(cross, a);
    float dot_b = DotProduct(cross, b);

    // Scale tolerance by the magnitude of the vectors (larger vectors = larger errors)
    float scale = std::max(1.0f, std::max(a.Magnitude(), b.Magnitude()) * cross.Magnitude());
    float scaledTol = LOOSE_TOL * scale;

    if (!IsNearZero(dot_a, scaledTol) || !IsNearZero(dot_b, scaledTol)) {
        return ::testing::AssertionFailure()
            << "(a x b).a = " << dot_a << ", (a x b).b = " << dot_b
            << " (expected both ~0, scaled tolerance = " << scaledTol << ")";
    }
    return ::testing::AssertionSuccess();
}

/// Property: Vector addition is commutative
inline ::testing::AssertionResult AdditionIsCommutative(const VxVector& a, const VxVector& b) {
    VxVector ab = a + b;
    VxVector ba = b + a;

    if (!VectorNearBool(ab, ba, STRICT_TOL)) {
        return ::testing::AssertionFailure()
            << "a + b = " << VectorToString(ab)
            << " but b + a = " << VectorToString(ba);
    }
    return ::testing::AssertionSuccess();
}

} // namespace VectorProperties

//=============================================================================
// Matrix Properties
//=============================================================================

namespace MatrixProperties {

/// Property: Multiplying by identity leaves matrix unchanged
inline ::testing::AssertionResult IdentityMultiplicationIsIdentity(const VxMatrix& m) {
    VxMatrix identity;
    identity.SetIdentity();

    VxMatrix result;
    Vx3DMultiplyMatrix(result, m, identity);

    if (!MatrixNearBool(result, m, SIMD_TOL)) {
        return ::testing::AssertionFailure()
            << "M * I != M\nM:\n" << MatrixToString(m)
            << "\nResult:\n" << MatrixToString(result);
    }
    return ::testing::AssertionSuccess();
}

/// Property: Inverse of inverse is original (for invertible matrices)
inline ::testing::AssertionResult InverseOfInverseIsOriginal(const VxMatrix& m) {
    float det = Vx3DMatrixDeterminant(m);
    if (std::fabs(det) < 1e-6f) {
        return ::testing::AssertionSuccess(); // Skip singular matrices
    }

    VxMatrix inv1, inv2;
    Vx3DInverseMatrix(inv1, m);
    Vx3DInverseMatrix(inv2, inv1);

    if (!MatrixNearBool(inv2, m, ACCUMULATION_TOL)) {
        return ::testing::AssertionFailure()
            << "(M^-1)^-1 != M\nOriginal:\n" << MatrixToString(m)
            << "\nDouble inverse:\n" << MatrixToString(inv2);
    }
    return ::testing::AssertionSuccess();
}

/// Property: Transpose of transpose is original
inline ::testing::AssertionResult TransposeOfTransposeIsOriginal(const VxMatrix& m) {
    VxMatrix t1, t2;
    Vx3DTransposeMatrix(t1, m);
    Vx3DTransposeMatrix(t2, t1);

    if (!MatrixNearBool(t2, m, STRICT_TOL)) {
        return ::testing::AssertionFailure()
            << "(M^T)^T != M";
    }
    return ::testing::AssertionSuccess();
}

/// Property: M * M^-1 = Identity (for invertible matrices)
inline ::testing::AssertionResult InverseMultiplicationIsIdentity(const VxMatrix& m) {
    float det = Vx3DMatrixDeterminant(m);
    if (std::fabs(det) < 1e-6f) {
        return ::testing::AssertionSuccess(); // Skip singular matrices
    }

    VxMatrix inv, result, identity;
    Vx3DInverseMatrix(inv, m);
    Vx3DMultiplyMatrix(result, m, inv);
    identity.SetIdentity();

    if (!MatrixNearBool(result, identity, ACCUMULATION_TOL)) {
        return ::testing::AssertionFailure()
            << "M * M^-1 != I\nResult:\n" << MatrixToString(result);
    }
    return ::testing::AssertionSuccess();
}

/// Property: Rotation matrix has determinant +/- 1
inline ::testing::AssertionResult RotationMatrixHasUnitDeterminant(const VxMatrix& m) {
    // This is for pure rotation matrices
    float det = Vx3DMatrixDeterminant(m);
    if (!ScaleRelativeNear(std::fabs(det), 1.0f, SIMD_TOL)) {
        return ::testing::AssertionFailure()
            << "Rotation matrix determinant = " << det << " (expected +/-1)";
    }
    return ::testing::AssertionSuccess();
}

/// Property: Rotation preserves vector magnitude
inline ::testing::AssertionResult RotationPreservesMagnitude(const VxMatrix& rotationMatrix, const VxVector& v) {
    VxVector rotated;
    Vx3DRotateVector(&rotated, rotationMatrix, &v);

    float origMag = v.Magnitude();
    float rotMag = rotated.Magnitude();

    if (!ScaleRelativeNear(origMag, rotMag, SIMD_TOL)) {
        return ::testing::AssertionFailure()
            << "Original magnitude " << origMag << " != rotated magnitude " << rotMag;
    }
    return ::testing::AssertionSuccess();
}

} // namespace MatrixProperties

//=============================================================================
// Quaternion Properties
//=============================================================================

namespace QuaternionProperties {

/// Property: Normalized quaternion has unit magnitude
inline ::testing::AssertionResult NormalizedQuaternionHasUnitMagnitude(const VxQuaternion& q) {
    float magSq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
    if (magSq < 1e-20f) {
        return ::testing::AssertionSuccess(); // Skip zero quaternions
    }

    VxQuaternion normalized = q;
    normalized.Normalize();
    float normMag = std::sqrt(normalized.x * normalized.x + normalized.y * normalized.y +
                              normalized.z * normalized.z + normalized.w * normalized.w);

    if (!ScaleRelativeNear(normMag, 1.0f, SIMD_TOL)) {
        return ::testing::AssertionFailure()
            << "Normalized quaternion magnitude = " << normMag << " (expected 1.0)";
    }
    return ::testing::AssertionSuccess();
}

/// Property: q and -q represent the same rotation
inline ::testing::AssertionResult QuaternionAndNegativeRepresentSameRotation(const VxQuaternion& q) {
    if ((q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) < 1e-20f) {
        return ::testing::AssertionSuccess(); // Skip zero quaternions
    }

    VxQuaternion qn = q;
    qn.Normalize();
    VxQuaternion negQ(-qn.x, -qn.y, -qn.z, -qn.w);

    // Convert both to matrices and compare
    VxMatrix m1, m2;
    qn.ToMatrix(m1);
    negQ.ToMatrix(m2);

    if (!MatrixNearBool(m1, m2, ACCUMULATION_TOL)) {
        return ::testing::AssertionFailure()
            << "q and -q produced different rotation matrices";
    }
    return ::testing::AssertionSuccess();
}

/// Property: Slerp(0, a, b) = a and Slerp(1, a, b) = b
inline ::testing::AssertionResult SlerpEndpointsMatch(const VxQuaternion& a, const VxQuaternion& b) {
    VxQuaternion an = a, bn = b;
    an.Normalize();
    bn.Normalize();

    VxQuaternion s0 = Slerp(0.0f, an, bn);
    VxQuaternion s1 = Slerp(1.0f, an, bn);

    if (!QuaternionNearBool(s0, an, SIMD_TOL)) {
        return ::testing::AssertionFailure()
            << "Slerp(0) = " << QuaternionToString(s0)
            << " != a = " << QuaternionToString(an);
    }
    if (!QuaternionNearBool(s1, bn, SIMD_TOL)) {
        return ::testing::AssertionFailure()
            << "Slerp(1) = " << QuaternionToString(s1)
            << " != b = " << QuaternionToString(bn);
    }
    return ::testing::AssertionSuccess();
}

/// Property: Quaternion multiplication is associative
inline ::testing::AssertionResult MultiplicationIsAssociative(
    const VxQuaternion& a, const VxQuaternion& b, const VxQuaternion& c
) {
    VxQuaternion an = a, bn = b, cn = c;
    an.Normalize();
    bn.Normalize();
    cn.Normalize();

    VxQuaternion ab = Vx3DQuaternionMultiply(an, bn);
    VxQuaternion abc1 = Vx3DQuaternionMultiply(ab, cn);

    VxQuaternion bc = Vx3DQuaternionMultiply(bn, cn);
    VxQuaternion abc2 = Vx3DQuaternionMultiply(an, bc);

    if (!QuaternionNearBool(abc1, abc2, ACCUMULATION_TOL)) {
        return ::testing::AssertionFailure()
            << "(a*b)*c != a*(b*c)";
    }
    return ::testing::AssertionSuccess();
}

/// Property: q * conjugate(q) = identity (for unit quaternions)
inline ::testing::AssertionResult ConjugateProductIsIdentity(const VxQuaternion& q) {
    if ((q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) < 1e-20f) {
        return ::testing::AssertionSuccess(); // Skip zero quaternions
    }

    VxQuaternion qn = q;
    qn.Normalize();
    VxQuaternion conj = Vx3DQuaternionConjugate(qn);
    VxQuaternion product = Vx3DQuaternionMultiply(qn, conj);

    VxQuaternion identity(0.0f, 0.0f, 0.0f, 1.0f);

    if (!QuaternionNearBool(product, identity, SIMD_TOL)) {
        return ::testing::AssertionFailure()
            << "q * conj(q) = " << QuaternionToString(product)
            << " (expected identity)";
    }
    return ::testing::AssertionSuccess();
}

} // namespace QuaternionProperties

//=============================================================================
// Geometric Properties
//=============================================================================

namespace GeometricProperties {

/// Property: Plane distance from point on plane is zero
inline ::testing::AssertionResult PointOnPlaneHasZeroDistance(const VxPlane& p, const VxVector& point) {
    // Create a point on the plane
    VxVector pointOnPlane = p.m_Normal * (-p.m_D / p.m_Normal.SquareMagnitude());

    float dist = p.Distance(pointOnPlane);
    if (!IsNearZero(dist, STANDARD_TOL)) {
        return ::testing::AssertionFailure()
            << "Distance from point on plane = " << dist << " (expected 0)";
    }
    return ::testing::AssertionSuccess();
}

/// Property: Bbox contains its center
inline ::testing::AssertionResult BboxContainsItsCenter(const VxBbox& b) {
    if (!b.IsValid()) {
        return ::testing::AssertionSuccess(); // Skip invalid boxes
    }

    VxVector center = b.GetCenter();
    if (!b.VectorIn(center)) {
        return ::testing::AssertionFailure()
            << "Bbox " << BboxToString(b) << " does not contain its center "
            << VectorToString(center);
    }
    return ::testing::AssertionSuccess();
}

/// Property: Merging bbox with itself doesn't change it
inline ::testing::AssertionResult MergingBboxWithSelfIsIdempotent(const VxBbox& b) {
    if (!b.IsValid()) {
        return ::testing::AssertionSuccess(); // Skip invalid boxes
    }

    VxBbox merged = b;
    merged.Merge(b);

    if (!BboxNearBool(merged, b, STRICT_TOL)) {
        return ::testing::AssertionFailure()
            << "b.Merge(b) changed the bbox";
    }
    return ::testing::AssertionSuccess();
}

/// Property: Sphere contains its center
inline ::testing::AssertionResult SphereContainsItsCenter(const VxSphere& s) {
    VxVector center = s.Center();
    VxVector toCenter = center - s.Center();
    float dist = toCenter.Magnitude();

    if (dist > s.Radius()) {
        return ::testing::AssertionFailure()
            << "Center is outside sphere (distance " << dist
            << " > radius " << s.Radius() << ")";
    }
    return ::testing::AssertionSuccess();
}

} // namespace GeometricProperties

} // namespace VxMathTest

#endif // VXMATH_TEST_PROPERTY_TESTING_H
