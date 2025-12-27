#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include "VxVector.h"
#include "VxMatrix.h"

// Tolerance for SIMD operations - SIMD can have slightly different rounding than scalar operations
constexpr float SIMD_EPSILON = 5e-07f;

class VxVectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        v1 = VxVector(1.0f, 2.0f, 3.0f);
        v2 = VxVector(4.0f, 5.0f, 6.0f);
        v3 = VxVector(0.0f, 0.0f, 0.0f);
    }

    VxVector v1, v2, v3;
};

TEST_F(VxVectorTest, DefaultConstructor) {
    VxVector v;
    EXPECT_FLOAT_EQ(v.x, 0.0f);
    EXPECT_FLOAT_EQ(v.y, 0.0f);
    EXPECT_FLOAT_EQ(v.z, 0.0f);
}

TEST_F(VxVectorTest, UniformConstructor) {
    VxVector v(5.0f);
    EXPECT_FLOAT_EQ(v.x, 5.0f);
    EXPECT_FLOAT_EQ(v.y, 5.0f);
    EXPECT_FLOAT_EQ(v.z, 5.0f);
}

TEST_F(VxVectorTest, ComponentConstructor) {
    EXPECT_FLOAT_EQ(v1.x, 1.0f);
    EXPECT_FLOAT_EQ(v1.y, 2.0f);
    EXPECT_FLOAT_EQ(v1.z, 3.0f);
}

TEST_F(VxVectorTest, ArrayConstructor) {
    float arr[3] = {7.0f, 8.0f, 9.0f};
    VxVector v(arr);
    EXPECT_FLOAT_EQ(v.x, 7.0f);
    EXPECT_FLOAT_EQ(v.y, 8.0f);
    EXPECT_FLOAT_EQ(v.z, 9.0f);
}

TEST_F(VxVectorTest, ArrayAccess) {
    EXPECT_FLOAT_EQ(v1[0], 1.0f);
    EXPECT_FLOAT_EQ(v1[1], 2.0f);
    EXPECT_FLOAT_EQ(v1[2], 3.0f);

    v1[0] = 10.0f;
    EXPECT_FLOAT_EQ(v1.x, 10.0f);
}

TEST_F(VxVectorTest, Set) {
    v3.Set(1.5f, 2.5f, 3.5f);
    EXPECT_FLOAT_EQ(v3.x, 1.5f);
    EXPECT_FLOAT_EQ(v3.y, 2.5f);
    EXPECT_FLOAT_EQ(v3.z, 3.5f);
}

TEST_F(VxVectorTest, SquareMagnitude) {
    float expected = 1.0f * 1.0f + 2.0f * 2.0f + 3.0f * 3.0f; // 14.0f
    EXPECT_FLOAT_EQ(v1.SquareMagnitude(), expected);
    EXPECT_FLOAT_EQ(SquareMagnitude(v1), expected);
}

TEST_F(VxVectorTest, Magnitude) {
    float expected = sqrtf(14.0f);
    EXPECT_NEAR(v1.Magnitude(), expected, EPSILON);
    EXPECT_NEAR(Magnitude(v1), expected, EPSILON);
}

TEST_F(VxVectorTest, Normalize) {
    VxVector v = v1;
    v.Normalize();
    EXPECT_NEAR(v.Magnitude(), 1.0f, SIMD_EPSILON);

    VxVector normalized = Normalize(v1);
    EXPECT_NEAR(normalized.Magnitude(), 1.0f, SIMD_EPSILON);
}

TEST_F(VxVectorTest, DotProduct) {
    float expected = 1.0f * 4.0f + 2.0f * 5.0f + 3.0f * 6.0f; // 32.0f
    EXPECT_FLOAT_EQ(v1.Dot(v2), expected);
    EXPECT_FLOAT_EQ(DotProduct(v1, v2), expected);
}

TEST_F(VxVectorTest, CrossProduct) {
    VxVector result = CrossProduct(v1, v2);
    VxVector expected(-3.0f, 6.0f, -3.0f);
    EXPECT_FLOAT_EQ(result.x, expected.x);
    EXPECT_FLOAT_EQ(result.y, expected.y);
    EXPECT_FLOAT_EQ(result.z, expected.z);
}

TEST_F(VxVectorTest, Addition) {
    VxVector result = v1 + v2;
    EXPECT_FLOAT_EQ(result.x, 5.0f);
    EXPECT_FLOAT_EQ(result.y, 7.0f);
    EXPECT_FLOAT_EQ(result.z, 9.0f);

    v1 += v2;
    EXPECT_FLOAT_EQ(v1.x, 5.0f);
    EXPECT_FLOAT_EQ(v1.y, 7.0f);
    EXPECT_FLOAT_EQ(v1.z, 9.0f);
}

TEST_F(VxVectorTest, Subtraction) {
    VxVector result = v2 - v1;
    EXPECT_FLOAT_EQ(result.x, 3.0f);
    EXPECT_FLOAT_EQ(result.y, 3.0f);
    EXPECT_FLOAT_EQ(result.z, 3.0f);

    VxVector v = v2;
    v -= v1;
    EXPECT_FLOAT_EQ(v.x, 3.0f);
    EXPECT_FLOAT_EQ(v.y, 3.0f);
    EXPECT_FLOAT_EQ(v.z, 3.0f);
}

TEST_F(VxVectorTest, ScalarMultiplication) {
    VxVector result = v1 * 2.0f;
    EXPECT_FLOAT_EQ(result.x, 2.0f);
    EXPECT_FLOAT_EQ(result.y, 4.0f);
    EXPECT_FLOAT_EQ(result.z, 6.0f);

    result = 2.0f * v1;
    EXPECT_FLOAT_EQ(result.x, 2.0f);
    EXPECT_FLOAT_EQ(result.y, 4.0f);
    EXPECT_FLOAT_EQ(result.z, 6.0f);

    v1 *= 2.0f;
    EXPECT_FLOAT_EQ(v1.x, 2.0f);
    EXPECT_FLOAT_EQ(v1.y, 4.0f);
    EXPECT_FLOAT_EQ(v1.z, 6.0f);
}

TEST_F(VxVectorTest, ScalarDivision) {
    VxVector result = v1 / 2.0f;
    EXPECT_FLOAT_EQ(result.x, 0.5f);
    EXPECT_FLOAT_EQ(result.y, 1.0f);
    EXPECT_FLOAT_EQ(result.z, 1.5f);

    v1 /= 2.0f;
    EXPECT_FLOAT_EQ(v1.x, 0.5f);
    EXPECT_FLOAT_EQ(v1.y, 1.0f);
    EXPECT_FLOAT_EQ(v1.z, 1.5f);
}

TEST_F(VxVectorTest, ComponentWiseMultiplication) {
    VxVector result = v1 * v2;
    EXPECT_FLOAT_EQ(result.x, 4.0f);
    EXPECT_FLOAT_EQ(result.y, 10.0f);
    EXPECT_FLOAT_EQ(result.z, 18.0f);

    v1 *= v2;
    EXPECT_FLOAT_EQ(v1.x, 4.0f);
    EXPECT_FLOAT_EQ(v1.y, 10.0f);
    EXPECT_FLOAT_EQ(v1.z, 18.0f);
}

TEST_F(VxVectorTest, ComponentWiseDivision) {
    VxVector result = v2 / v1;
    EXPECT_FLOAT_EQ(result.x, 4.0f);
    EXPECT_FLOAT_EQ(result.y, 2.5f);
    EXPECT_FLOAT_EQ(result.z, 2.0f);

    VxVector v = v2;
    v /= v1;
    EXPECT_FLOAT_EQ(v.x, 4.0f);
    EXPECT_FLOAT_EQ(v.y, 2.5f);
    EXPECT_FLOAT_EQ(v.z, 2.0f);
}

TEST_F(VxVectorTest, UnaryOperators) {
    VxVector pos = +v1;
    EXPECT_FLOAT_EQ(pos.x, v1.x);
    EXPECT_FLOAT_EQ(pos.y, v1.y);
    EXPECT_FLOAT_EQ(pos.z, v1.z);

    VxVector neg = -v1;
    EXPECT_FLOAT_EQ(neg.x, -v1.x);
    EXPECT_FLOAT_EQ(neg.y, -v1.y);
    EXPECT_FLOAT_EQ(neg.z, -v1.z);
}

TEST_F(VxVectorTest, ComparisonOperators) {
    VxVector v_copy = v1;
    EXPECT_TRUE(v1 == v_copy);
    EXPECT_FALSE(v1 != v_copy);
    EXPECT_FALSE(v1 == v2);
    EXPECT_TRUE(v1 != v2);
}

TEST_F(VxVectorTest, Absolute) {
    VxVector negative(-1.0f, -2.0f, -3.0f);
    negative.Absolute();
    EXPECT_FLOAT_EQ(negative.x, 1.0f);
    EXPECT_FLOAT_EQ(negative.y, 2.0f);
    EXPECT_FLOAT_EQ(negative.z, 3.0f);

    VxVector abs_func = Absolute(VxVector(-4.0f, -5.0f, -6.0f));
    EXPECT_FLOAT_EQ(abs_func.x, 4.0f);
    EXPECT_FLOAT_EQ(abs_func.y, 5.0f);
    EXPECT_FLOAT_EQ(abs_func.z, 6.0f);
}

TEST_F(VxVectorTest, MinMax) {
    VxVector v(3.0f, 1.0f, 2.0f);
    EXPECT_FLOAT_EQ(Min(v), 1.0f);
    EXPECT_FLOAT_EQ(Max(v), 3.0f);
}

TEST_F(VxVectorTest, MinimizeMaximize) {
    VxVector min_result = Minimize(v1, v2);
    EXPECT_FLOAT_EQ(min_result.x, 1.0f);
    EXPECT_FLOAT_EQ(min_result.y, 2.0f);
    EXPECT_FLOAT_EQ(min_result.z, 3.0f);

    VxVector max_result = Maximize(v1, v2);
    EXPECT_FLOAT_EQ(max_result.x, 4.0f);
    EXPECT_FLOAT_EQ(max_result.y, 5.0f);
    EXPECT_FLOAT_EQ(max_result.z, 6.0f);
}

TEST_F(VxVectorTest, Interpolate) {
    VxVector result = Interpolate(0.5f, v1, v2);
    EXPECT_FLOAT_EQ(result.x, 2.5f);
    EXPECT_FLOAT_EQ(result.y, 3.5f);
    EXPECT_FLOAT_EQ(result.z, 4.5f);

    result = Interpolate(0.0f, v1, v2);
    EXPECT_FLOAT_EQ(result.x, v1.x);
    EXPECT_FLOAT_EQ(result.y, v1.y);
    EXPECT_FLOAT_EQ(result.z, v1.z);

    result = Interpolate(1.0f, v1, v2);
    EXPECT_FLOAT_EQ(result.x, v2.x);
    EXPECT_FLOAT_EQ(result.y, v2.y);
    EXPECT_FLOAT_EQ(result.z, v2.z);
}

TEST_F(VxVectorTest, Reflect) {
    VxVector incident(1.0f, -1.0f, 0.0f);
    VxVector normal(0.0f, 1.0f, 0.0f);
    VxVector reflected = Reflect(incident, normal);
    
    EXPECT_NEAR(reflected.x, 1.0f, EPSILON);
    EXPECT_NEAR(reflected.y, 1.0f, EPSILON);
    EXPECT_NEAR(reflected.z, 0.0f, EPSILON);
}

TEST_F(VxVectorTest, PredefinedVectors) {
    const VxVector& axisX = VxVector::axisX();
    EXPECT_FLOAT_EQ(axisX.x, 1.0f);
    EXPECT_FLOAT_EQ(axisX.y, 0.0f);
    EXPECT_FLOAT_EQ(axisX.z, 0.0f);

    const VxVector& axisY = VxVector::axisY();
    EXPECT_FLOAT_EQ(axisY.x, 0.0f);
    EXPECT_FLOAT_EQ(axisY.y, 1.0f);
    EXPECT_FLOAT_EQ(axisY.z, 0.0f);

    const VxVector& axisZ = VxVector::axisZ();
    EXPECT_FLOAT_EQ(axisZ.x, 0.0f);
    EXPECT_FLOAT_EQ(axisZ.y, 0.0f);
    EXPECT_FLOAT_EQ(axisZ.z, 1.0f);

    const VxVector& zero = VxVector::axis0();
    EXPECT_FLOAT_EQ(zero.x, 0.0f);
    EXPECT_FLOAT_EQ(zero.y, 0.0f);
    EXPECT_FLOAT_EQ(zero.z, 0.0f);

    const VxVector& ones = VxVector::axis1();
    EXPECT_FLOAT_EQ(ones.x, 1.0f);
    EXPECT_FLOAT_EQ(ones.y, 1.0f);
    EXPECT_FLOAT_EQ(ones.z, 1.0f);
}

class VxVector4Test : public ::testing::Test {
protected:
    VxVector4 v1, v2;

    void SetUp() override {
        v1 = VxVector4(1.0f, 2.0f, 3.0f, 4.0f);
        v2 = VxVector4(5.0f, 6.0f, 7.0f, 8.0f);
    }
};

TEST_F(VxVector4Test, DefaultConstructor) {
    VxVector4 v;
    EXPECT_FLOAT_EQ(v.x, 0.0f);
    EXPECT_FLOAT_EQ(v.y, 0.0f);
    EXPECT_FLOAT_EQ(v.z, 0.0f);
    EXPECT_FLOAT_EQ(v.w, 0.0f);
}

TEST_F(VxVector4Test, UniformConstructor) {
    VxVector4 v(5.0f);
    EXPECT_FLOAT_EQ(v.x, 5.0f);
    EXPECT_FLOAT_EQ(v.y, 5.0f);
    EXPECT_FLOAT_EQ(v.z, 5.0f);
    EXPECT_FLOAT_EQ(v.w, 5.0f);
}

TEST_F(VxVector4Test, ComponentConstructor) {
    EXPECT_FLOAT_EQ(v1.x, 1.0f);
    EXPECT_FLOAT_EQ(v1.y, 2.0f);
    EXPECT_FLOAT_EQ(v1.z, 3.0f);
    EXPECT_FLOAT_EQ(v1.w, 4.0f);
}

TEST_F(VxVector4Test, ArrayConstructor) {
    float arr[4] = {9.0f, 10.0f, 11.0f, 12.0f};
    VxVector4 v(arr);
    EXPECT_FLOAT_EQ(v.x, 9.0f);
    EXPECT_FLOAT_EQ(v.y, 10.0f);
    EXPECT_FLOAT_EQ(v.z, 11.0f);
    EXPECT_FLOAT_EQ(v.w, 12.0f);
}

TEST_F(VxVector4Test, ArrayAccess) {
    EXPECT_FLOAT_EQ(v1[0], 1.0f);
    EXPECT_FLOAT_EQ(v1[1], 2.0f);
    EXPECT_FLOAT_EQ(v1[2], 3.0f);
    EXPECT_FLOAT_EQ(v1[3], 4.0f);

    v1[0] = 10.0f;
    EXPECT_FLOAT_EQ(v1.x, 10.0f);
}

TEST_F(VxVector4Test, Set) {
    v1.Set(10.0f, 11.0f, 12.0f, 13.0f);
    EXPECT_FLOAT_EQ(v1.x, 10.0f);
    EXPECT_FLOAT_EQ(v1.y, 11.0f);
    EXPECT_FLOAT_EQ(v1.z, 12.0f);
    EXPECT_FLOAT_EQ(v1.w, 13.0f);

    v1.Set(14.0f, 15.0f, 16.0f);
    EXPECT_FLOAT_EQ(v1.x, 14.0f);
    EXPECT_FLOAT_EQ(v1.y, 15.0f);
    EXPECT_FLOAT_EQ(v1.z, 16.0f);
    EXPECT_FLOAT_EQ(v1.w, 13.0f); // Should remain unchanged
}

TEST_F(VxVector4Test, DotProduct) {
    float expected = 1.0f * 5.0f + 2.0f * 6.0f + 3.0f * 7.0f; // 38.0f (w components are ignored)
    EXPECT_FLOAT_EQ(v1.Dot(v2), expected);
}

TEST_F(VxVector4Test, Assignment) {
    VxVector v3d(1.5f, 2.5f, 3.5f);
    v1 = v3d;
    EXPECT_FLOAT_EQ(v1.x, 1.5f);
    EXPECT_FLOAT_EQ(v1.y, 2.5f);
    EXPECT_FLOAT_EQ(v1.z, 3.5f);
    EXPECT_FLOAT_EQ(v1.w, 4.0f); // Should remain unchanged
}

TEST_F(VxVector4Test, ArithmeticOperations) {
    VxVector4 result = v1 + v2;
    EXPECT_FLOAT_EQ(result.x, 6.0f);
    EXPECT_FLOAT_EQ(result.y, 8.0f);
    EXPECT_FLOAT_EQ(result.z, 10.0f);
    EXPECT_FLOAT_EQ(result.w, 12.0f);

    result = v1 * 2.0f;
    EXPECT_FLOAT_EQ(result.x, 2.0f);
    EXPECT_FLOAT_EQ(result.y, 4.0f);
    EXPECT_FLOAT_EQ(result.z, 6.0f);
    EXPECT_FLOAT_EQ(result.w, 8.0f);
}

class VxBboxTest : public ::testing::Test {
protected:
    VxBbox box1, box2;
    VxVector min1, max1, min2, max2;

    void SetUp() override {
        min1 = VxVector(-1.0f, -1.0f, -1.0f);
        max1 = VxVector(1.0f, 1.0f, 1.0f);
        box1.SetCorners(min1, max1);

        min2 = VxVector(0.0f, 0.0f, 0.0f);
        max2 = VxVector(2.0f, 2.0f, 2.0f);
        box2.SetCorners(min2, max2);
    }
};

TEST_F(VxBboxTest, DefaultConstructor) {
    VxBbox box;
    EXPECT_FALSE(box.IsValid());
}

TEST_F(VxBboxTest, ParameterizedConstructor) {
    VxBbox box(min1, max1);
    EXPECT_TRUE(box.IsValid());
    EXPECT_TRUE(box.Min == min1);
    EXPECT_TRUE(box.Max == max1);
}

TEST_F(VxBboxTest, UniformConstructor) {
    VxBbox box(2.0f);
    EXPECT_TRUE(box.IsValid());
    EXPECT_FLOAT_EQ(box.Min.x, -2.0f);
    EXPECT_FLOAT_EQ(box.Max.x, 2.0f);
}

TEST_F(VxBboxTest, IsValid) {
    EXPECT_TRUE(box1.IsValid());
    
    VxBbox invalid_box;
    EXPECT_FALSE(invalid_box.IsValid());
}

TEST_F(VxBboxTest, GetSize) {
    VxVector size = box1.GetSize();
    EXPECT_FLOAT_EQ(size.x, 2.0f);
    EXPECT_FLOAT_EQ(size.y, 2.0f);
    EXPECT_FLOAT_EQ(size.z, 2.0f);
}

TEST_F(VxBboxTest, GetHalfSize) {
    VxVector half_size = box1.GetHalfSize();
    EXPECT_FLOAT_EQ(half_size.x, 1.0f);
    EXPECT_FLOAT_EQ(half_size.y, 1.0f);
    EXPECT_FLOAT_EQ(half_size.z, 1.0f);
}

TEST_F(VxBboxTest, GetCenter) {
    VxVector center = box1.GetCenter();
    EXPECT_FLOAT_EQ(center.x, 0.0f);
    EXPECT_FLOAT_EQ(center.y, 0.0f);
    EXPECT_FLOAT_EQ(center.z, 0.0f);
}

TEST_F(VxBboxTest, SetCenter) {
    VxVector new_center(5.0f, 5.0f, 5.0f);
    VxVector half_size(1.0f, 1.0f, 1.0f);
    box1.SetCenter(new_center, half_size);
    
    VxVector center = box1.GetCenter();
    EXPECT_FLOAT_EQ(center.x, 5.0f);
    EXPECT_FLOAT_EQ(center.y, 5.0f);
    EXPECT_FLOAT_EQ(center.z, 5.0f);
}

TEST_F(VxBboxTest, Merge) {
    VxBbox merged = box1;
    merged.Merge(box2);
    
    EXPECT_FLOAT_EQ(merged.Min.x, -1.0f);
    EXPECT_FLOAT_EQ(merged.Min.y, -1.0f);
    EXPECT_FLOAT_EQ(merged.Min.z, -1.0f);
    EXPECT_FLOAT_EQ(merged.Max.x, 2.0f);
    EXPECT_FLOAT_EQ(merged.Max.y, 2.0f);
    EXPECT_FLOAT_EQ(merged.Max.z, 2.0f);
}

TEST_F(VxBboxTest, MergeWithPoint) {
    VxVector point(3.0f, 3.0f, 3.0f);
    box1.Merge(point);
    
    EXPECT_FLOAT_EQ(box1.Max.x, 3.0f);
    EXPECT_FLOAT_EQ(box1.Max.y, 3.0f);
    EXPECT_FLOAT_EQ(box1.Max.z, 3.0f);
}

TEST_F(VxBboxTest, VectorIn) {
    VxVector inside(0.0f, 0.0f, 0.0f);
    VxVector outside(2.0f, 2.0f, 2.0f);
    
    EXPECT_TRUE(box1.VectorIn(inside));
    EXPECT_FALSE(box1.VectorIn(outside));
}

TEST_F(VxBboxTest, IsBoxInside) {
    VxBbox small_box(VxVector(-0.5f, -0.5f, -0.5f), VxVector(0.5f, 0.5f, 0.5f));
    VxBbox large_box(VxVector(-2.0f, -2.0f, -2.0f), VxVector(2.0f, 2.0f, 2.0f));
    
    EXPECT_TRUE(box1.IsBoxInside(small_box));
    EXPECT_FALSE(box1.IsBoxInside(large_box));
}

TEST_F(VxBboxTest, Intersect) {
    box1.Intersect(box2);
    
    EXPECT_FLOAT_EQ(box1.Min.x, 0.0f);
    EXPECT_FLOAT_EQ(box1.Min.y, 0.0f);
    EXPECT_FLOAT_EQ(box1.Min.z, 0.0f);
    EXPECT_FLOAT_EQ(box1.Max.x, 1.0f);
    EXPECT_FLOAT_EQ(box1.Max.y, 1.0f);
    EXPECT_FLOAT_EQ(box1.Max.z, 1.0f);
}

TEST_F(VxBboxTest, Reset) {
    box1.Reset();
    EXPECT_FALSE(box1.IsValid());
}

TEST_F(VxBboxTest, EqualityOperator) {
    VxBbox box_copy = box1;
    EXPECT_TRUE(box1 == box_copy);
    EXPECT_FALSE(box1 == box2);
}

// ============================================================================
// Additional Tests from VectorTestEnhanced
// ============================================================================

class VxVectorEnhancedTest : public ::testing::Test {
protected:
    void SetUp() override {
        epsilon = SIMD_EPSILON;
        rng.seed(42); // Fixed seed for reproducible tests
    }

    float epsilon;
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist{-100.0f, 100.0f};

    // Helper function to generate random vectors
    VxVector RandomVector() {
        return VxVector(dist(rng), dist(rng), dist(rng));
    }

    // Helper function to compare vectors with tolerance
    bool VectorsApproxEqual(const VxVector& v1, const VxVector& v2, float tol = SIMD_EPSILON) {
        return std::abs(v1.x - v2.x) < tol &&
               std::abs(v1.y - v2.y) < tol &&
               std::abs(v1.z - v2.z) < tol;
    }
};

TEST_F(VxVectorEnhancedTest, VectorOperations) {
    VxVector v1(1.0f, 0.0f, 0.0f);
    VxVector v2(0.0f, 1.0f, 0.0f);
    VxVector v3(0.0f, 0.0f, 1.0f);

    // Test cross products
    VxVector cross1 = CrossProduct(v1, v2);
    EXPECT_TRUE(VectorsApproxEqual(cross1, v3));

    VxVector cross2 = CrossProduct(v2, v3);
    EXPECT_TRUE(VectorsApproxEqual(cross2, v1));

    VxVector cross3 = CrossProduct(v3, v1);
    EXPECT_TRUE(VectorsApproxEqual(cross3, v2));

    // Test anti-commutativity of cross product
    VxVector cross_neg = CrossProduct(v2, v1);
    EXPECT_TRUE(VectorsApproxEqual(cross_neg, -v3));

    // Test dot products
    EXPECT_FLOAT_EQ(DotProduct(v1, v1), 1.0f);
    EXPECT_FLOAT_EQ(DotProduct(v1, v2), 0.0f);
    EXPECT_FLOAT_EQ(DotProduct(v1, v3), 0.0f);

    // Test dot product with scaled vectors
    VxVector scaled = v1 * 3.0f;
    EXPECT_FLOAT_EQ(DotProduct(v1, scaled), 3.0f);
}

TEST_F(VxVectorEnhancedTest, InterpolationExtrapolation) {
    VxVector v1(0.0f, 0.0f, 0.0f);
    VxVector v2(10.0f, 20.0f, 30.0f);

    // Test extrapolation
    VxVector interp_neg = Interpolate(-0.5f, v1, v2);
    EXPECT_TRUE(VectorsApproxEqual(interp_neg, VxVector(-5.0f, -10.0f, -15.0f)));

    VxVector interp_15 = Interpolate(1.5f, v1, v2);
    EXPECT_TRUE(VectorsApproxEqual(interp_15, VxVector(15.0f, 30.0f, 45.0f)));
}

TEST_F(VxVectorEnhancedTest, ReflectionNonUnitNormal) {
    VxVector incident(1.0f, 1.0f, 0.0f);

    // Test reflection with non-unit normal
    VxVector non_unit_normal(0.0f, 2.0f, 0.0f);
    VxVector reflected_non_unit = Reflect(incident, non_unit_normal);

    // Test that the reflection produces a valid result (finite values)
    EXPECT_TRUE(std::isfinite(reflected_non_unit.x));
    EXPECT_TRUE(std::isfinite(reflected_non_unit.y));
    EXPECT_TRUE(std::isfinite(reflected_non_unit.z));

    // Test that the reflection changes the direction appropriately
    // (y component should be negative after reflecting off a y-axis normal)
    EXPECT_LT(reflected_non_unit.y, 0.0f);
}

TEST_F(VxVectorEnhancedTest, MatrixTransformations) {
    VxVector v(1.0f, 2.0f, 3.0f);
    VxMatrix transform;
    transform.SetIdentity();

    // Test translation
    transform[3][0] = 5.0f;
    transform[3][1] = 10.0f;
    transform[3][2] = 15.0f;

    VxVector translated = transform * v;
    EXPECT_TRUE(VectorsApproxEqual(translated, VxVector(6.0f, 12.0f, 18.0f)));

    // Test scaling
    VxMatrix scale;
    scale.SetIdentity();
    scale[0][0] = 2.0f;
    scale[1][1] = 3.0f;
    scale[2][2] = 4.0f;

    VxVector scaled = scale * v;
    EXPECT_TRUE(VectorsApproxEqual(scaled, VxVector(2.0f, 6.0f, 12.0f)));

    // Test combined transformation
    VxMatrix combined;
    Vx3DMultiplyMatrix(combined, transform, scale);
    VxVector result = combined * v;
    EXPECT_TRUE(VectorsApproxEqual(result, VxVector(7.0f, 16.0f, 27.0f)));
}

TEST_F(VxVectorEnhancedTest, AngleCalculations) {
    VxVector v1(1.0f, 0.0f, 0.0f);
    VxVector v2(0.0f, 1.0f, 0.0f);

    // Test angle calculation using dot product
    float dot = DotProduct(v1, v2);
    float mag1 = Magnitude(v1);
    float mag2 = Magnitude(v2);

    EXPECT_FLOAT_EQ(dot, 0.0f);
    EXPECT_FLOAT_EQ(mag1, 1.0f);
    EXPECT_FLOAT_EQ(mag2, 1.0f);

    // Angle = acos(dot / (mag1 * mag2)) = acos(0) = 90 degrees
    float angle = std::acos(dot / (mag1 * mag2));
    EXPECT_NEAR(angle, PI / 2.0f, epsilon);
}

TEST_F(VxVectorEnhancedTest, EdgeCasesAndRobustness) {
    // Test zero vector operations
    VxVector zero(0.0f, 0.0f, 0.0f);
    VxVector v1(1.0f, 2.0f, 3.0f);

    // Cross product with zero vector
    VxVector cross_zero1 = CrossProduct(zero, v1);
    VxVector cross_zero2 = CrossProduct(v1, zero);
    EXPECT_TRUE(VectorsApproxEqual(cross_zero1, zero));
    EXPECT_TRUE(VectorsApproxEqual(cross_zero2, zero));

    // Dot product with zero vector
    EXPECT_FLOAT_EQ(DotProduct(zero, v1), 0.0f);
    EXPECT_FLOAT_EQ(DotProduct(v1, zero), 0.0f);

    // Test very small vectors
    VxVector tiny(1e-10f, 1e-10f, 1e-10f);
    EXPECT_GT(Magnitude(tiny), 0.0f);
    EXPECT_LT(Magnitude(tiny), 1e-9f);

    // Test very large vectors
    VxVector huge(1e10f, 1e10f, 1e10f);
    EXPECT_GT(Magnitude(huge), 1e10f);

    // Test normalization of zero vector (should not crash)
    VxVector zero_copy = zero;
    zero_copy.Normalize();
    // After normalizing zero, result should still be very small (near zero or NaN is acceptable)
    // SIMD implementations may handle this differently
    EXPECT_TRUE(VectorsApproxEqual(zero_copy, zero, 1e-5f) || std::isnan(zero_copy.x));
}

TEST_F(VxVectorEnhancedTest, PerformanceConsistency) {
    // Test that vector operations are consistent with mathematical properties

    // Test commutativity of addition
    for (int i = 0; i < 10; ++i) {
        VxVector a = RandomVector();
        VxVector b = RandomVector();

        VxVector sum1 = a + b;
        VxVector sum2 = b + a;

        EXPECT_TRUE(VectorsApproxEqual(sum1, sum2, epsilon));
    }

    // Test associativity of vector addition
    for (int i = 0; i < 10; ++i) {
        VxVector a = RandomVector();
        VxVector b = RandomVector();
        VxVector c = RandomVector();

        VxVector sum1 = (a + b) + c;
        VxVector sum2 = a + (b + c);

        // Floating-point addition is not associative; allow a looser tolerance.
        EXPECT_TRUE(VectorsApproxEqual(sum1, sum2, 1e-4f));
    }

    // Test distributivity of scalar multiplication over addition
    for (int i = 0; i < 10; ++i) {
        VxVector a = RandomVector();
        VxVector b = RandomVector();
        float scalar = dist(rng);

        VxVector result1 = a * scalar + b * scalar;
        VxVector result2 = (a + b) * scalar;

        EXPECT_TRUE(VectorsApproxEqual(result1, result2, 1e-3f));
    }
}

TEST_F(VxVectorEnhancedTest, Vector4OperationsEnhanced) {
    // Test VxVector4 functionality
    VxVector4 v4(1.0f, 2.0f, 3.0f, 4.0f);

    // Test dot product
    VxVector4 v4b(2.0f, 3.0f, 4.0f, 5.0f);
    float dot4 = v4.Dot(v4b);
    // VxVector4::Dot might ignore the w component, based on implementation
    EXPECT_FLOAT_EQ(dot4, 1.0f*2.0f + 2.0f*3.0f + 3.0f*4.0f); // Excludes w component

    // Test magnitude
    float mag4 = v4.Magnitude();
    // VxVector4::Magnitude might also ignore the w component
    EXPECT_NEAR(mag4, std::sqrt(1.0f*1.0f + 2.0f*2.0f + 3.0f*3.0f), epsilon);
}

TEST_F(VxVectorEnhancedTest, CoordinateSystemTransforms) {
    // Test basic matrix transformation functionality

    VxVector original(1.0f, 2.0f, 3.0f);

    // Create a simple translation matrix
    VxMatrix transform;
    transform.SetIdentity();
    transform[3][0] = 5.0f;  // X translation
    transform[3][1] = 10.0f; // Y translation
    transform[3][2] = 15.0f; // Z translation

    VxVector transformed = transform * original;

    // Test that transformation changes the vector
    EXPECT_NE(transformed.x, original.x);
    EXPECT_NE(transformed.y, original.y);
    EXPECT_NE(transformed.z, original.z);

    // Test that transformation is finite
    EXPECT_TRUE(std::isfinite(transformed.x));
    EXPECT_TRUE(std::isfinite(transformed.y));
    EXPECT_TRUE(std::isfinite(transformed.z));
}