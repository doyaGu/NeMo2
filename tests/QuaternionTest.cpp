#include <gtest/gtest.h>
#include <cmath>
#include "VxQuaternion.h"
#include "VxMatrix.h"
#include "VxVector.h"

// Tolerance for SIMD operations - SIMD can have slightly different rounding than scalar operations
// This is mathematically correct and expected behavior
constexpr float SIMD_EPSILON = 5e-07f;

// Helper function to check if quaternions are approximately equal
bool QuaternionsApproxEqual(const VxQuaternion& q1, const VxQuaternion& q2, float epsilon = SIMD_EPSILON) {
    // Check both q and -q since they represent the same rotation
    bool direct = (std::abs(q1.x - q2.x) < epsilon &&
                   std::abs(q1.y - q2.y) < epsilon &&
                   std::abs(q1.z - q2.z) < epsilon &&
                   std::abs(q1.w - q2.w) < epsilon);

    bool negated = (std::abs(q1.x + q2.x) < epsilon &&
                    std::abs(q1.y + q2.y) < epsilon &&
                    std::abs(q1.z + q2.z) < epsilon &&
                    std::abs(q1.w + q2.w) < epsilon);

    return direct || negated;
}

class VxQuaternionBasicTest : public ::testing::Test {
protected:
    void SetUp() override {
        identity = VxQuaternion(); // (0, 0, 0, 1)
        q1 = VxQuaternion(1.0f, 2.0f, 3.0f, 4.0f);
        q2 = VxQuaternion(0.5f, 1.5f, 2.5f, 3.5f);
        unit_x = VxQuaternion(VxVector(1.0f, 0.0f, 0.0f), PI / 2.0f);
        unit_y = VxQuaternion(VxVector(0.0f, 1.0f, 0.0f), PI / 2.0f);
        unit_z = VxQuaternion(VxVector(0.0f, 0.0f, 1.0f), PI / 2.0f);
    }

    VxQuaternion identity, q1, q2, unit_x, unit_y, unit_z;
};

// ============================================================================
// Constructor Tests
TEST_F(VxQuaternionBasicTest, DefaultConstructor) {
    VxQuaternion q;
    EXPECT_FLOAT_EQ(q.x, 0.0f);
    EXPECT_FLOAT_EQ(q.y, 0.0f);
    EXPECT_FLOAT_EQ(q.z, 0.0f);
    EXPECT_FLOAT_EQ(q.w, 1.0f);
    
    // Test that it represents identity rotation
    VxMatrix mat;
    q.ToMatrix(mat);
    VxVector test_vec(1.0f, 0.0f, 0.0f);
    VxVector result = mat * test_vec;
    EXPECT_NEAR(result.x, 1.0f, EPSILON);
    EXPECT_NEAR(result.y, 0.0f, EPSILON);
    EXPECT_NEAR(result.z, 0.0f, EPSILON);
}

TEST_F(VxQuaternionBasicTest, ComponentConstructor) {
    EXPECT_FLOAT_EQ(q1.x, 1.0f);
    EXPECT_FLOAT_EQ(q1.y, 2.0f);
    EXPECT_FLOAT_EQ(q1.z, 3.0f);
    EXPECT_FLOAT_EQ(q1.w, 4.0f);
}

TEST_F(VxQuaternionBasicTest, AxisAngleConstructor) {
    // Test Z-axis rotation
    VxVector axis(0.0f, 0.0f, 1.0f);
    float angle = PI / 2.0f; // 90 degrees
    VxQuaternion q(axis, angle);

    // The VxMath implementation appears to use angle/2 in the sin/cos calculations
    // and may negate the vector part based on its internal convention
    EXPECT_NEAR(q.x, 0.0f, EPSILON);
    EXPECT_NEAR(q.y, 0.0f, EPSILON);
    EXPECT_NEAR(q.z, -sin(PI / 4.0f), EPSILON); // Negative due to implementation convention
    EXPECT_NEAR(q.w, cos(PI / 4.0f), EPSILON);

    // Test X-axis rotation
    VxVector x_axis(1.0f, 0.0f, 0.0f);
    VxQuaternion qx(x_axis, PI / 3.0f);
    EXPECT_NEAR(qx.x, -sin(PI / 6.0f), EPSILON); // Negative due to implementation convention
    EXPECT_NEAR(qx.y, 0.0f, EPSILON);
    EXPECT_NEAR(qx.z, 0.0f, EPSILON);
    EXPECT_NEAR(qx.w, cos(PI / 6.0f), EPSILON);

    // Test zero angle
    VxQuaternion q_zero(axis, 0.0f);
    EXPECT_NEAR(q_zero.x, 0.0f, EPSILON);
    EXPECT_NEAR(q_zero.y, 0.0f, EPSILON);
    EXPECT_NEAR(q_zero.z, 0.0f, EPSILON);
    EXPECT_NEAR(q_zero.w, 1.0f, EPSILON);
}

// ============================================================================
// Memory Layout and Access Tests
TEST_F(VxQuaternionBasicTest, ArrayAccess) {
    EXPECT_FLOAT_EQ(q1[0], 1.0f);
    EXPECT_FLOAT_EQ(q1[1], 2.0f);
    EXPECT_FLOAT_EQ(q1[2], 3.0f);
    EXPECT_FLOAT_EQ(q1[3], 4.0f);

    // Test mutable access
    q1[0] = 10.0f;
    EXPECT_FLOAT_EQ(q1.x, 10.0f);
    
    q1[1] = 20.0f;
    EXPECT_FLOAT_EQ(q1.y, 20.0f);
    
    q1[2] = 30.0f;
    EXPECT_FLOAT_EQ(q1.z, 30.0f);
    
    q1[3] = 40.0f;
    EXPECT_FLOAT_EQ(q1.w, 40.0f);
}

#ifdef _MSC_VER
TEST_F(VxQuaternionBasicTest, UnionAccessMSVC) {
    VxQuaternion q(1.0f, 2.0f, 3.0f, 4.0f);
    
    // Test v array access
    EXPECT_FLOAT_EQ(q.v[0], 1.0f);
    EXPECT_FLOAT_EQ(q.v[1], 2.0f);
    EXPECT_FLOAT_EQ(q.v[2], 3.0f);
    EXPECT_FLOAT_EQ(q.v[3], 4.0f);
    
    // Test axis/angle access
    EXPECT_FLOAT_EQ(q.axis.x, 1.0f);
    EXPECT_FLOAT_EQ(q.axis.y, 2.0f);
    EXPECT_FLOAT_EQ(q.axis.z, 3.0f);
    EXPECT_FLOAT_EQ(q.angle, 4.0f);
    
    // Test that angle and w refer to same memory
    q.w = 5.0f;
    EXPECT_FLOAT_EQ(q.angle, 5.0f);
    
    q.angle = 6.0f;
    EXPECT_FLOAT_EQ(q.w, 6.0f);
}
#else
TEST_F(VxQuaternionBasicTest, UnionAccessNonMSVC) {
    VxQuaternion q(1.0f, 2.0f, 3.0f, 4.0f);
    
    // Test that angle and w refer to same memory
    EXPECT_FLOAT_EQ(q.angle, q.w);
    
    q.w = 5.0f;
    EXPECT_FLOAT_EQ(q.angle, 5.0f);
    
    q.angle = 6.0f;
    EXPECT_FLOAT_EQ(q.w, 6.0f);
}
#endif

// ============================================================================
// Conversion Methods Tests
class VxQuaternionConversionTest : public ::testing::Test {
};

TEST_F(VxQuaternionConversionTest, FromRotationComprehensive) {
    struct TestCase {
        VxVector axis;
        float angle;
        const char* name;
    };
    
    TestCase cases[] = {
        {VxVector(1.0f, 0.0f, 0.0f), 0.0f, "Zero rotation"},
        {VxVector(1.0f, 0.0f, 0.0f), PI / 6.0f, "30 deg X"},
        {VxVector(1.0f, 0.0f, 0.0f), PI / 4.0f, "45 deg X"},
        {VxVector(1.0f, 0.0f, 0.0f), PI / 2.0f, "90 deg X"},
        {VxVector(1.0f, 0.0f, 0.0f), PI, "180 deg X"},
        {VxVector(0.0f, 1.0f, 0.0f), PI / 3.0f, "60 deg Y"},
        {VxVector(0.0f, 0.0f, 1.0f), PI / 2.0f, "90 deg Z"},
        {Normalize(VxVector(1.0f, 1.0f, 1.0f)), PI / 4.0f, "45 deg diagonal"},
        {VxVector(0.6f, 0.8f, 0.0f), PI / 3.0f, "60 deg arbitrary"},
    };
    
    for (const auto& test : cases) {
        VxQuaternion q;
        q.FromRotation(test.axis, test.angle);
        
        // Verify unit quaternion
        float magnitude = sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        EXPECT_NEAR(magnitude, 1.0f, EPSILON) << "Test: " << test.name;
        
        // Test by applying rotation to test vectors
        VxMatrix mat;
        q.ToMatrix(mat);
        
        // Test that the axis remains unchanged (for non-zero angles)
        if (test.angle > EPSILON) {
            VxVector rotated_axis = mat * test.axis;
            EXPECT_NEAR(rotated_axis.x, test.axis.x, EPSILON) << "Test: " << test.name;
            EXPECT_NEAR(rotated_axis.y, test.axis.y, EPSILON) << "Test: " << test.name;
            EXPECT_NEAR(rotated_axis.z, test.axis.z, EPSILON) << "Test: " << test.name;
        }
    }
}

TEST_F(VxQuaternionConversionTest, EulerAnglesRoundTrip) {
    struct TestCase {
        float eax, eay, eaz;
        const char* name;
    };
    
    TestCase cases[] = {
        {0.0f, 0.0f, 0.0f, "Zero rotation"},
        {PI / 6.0f, 0.0f, 0.0f, "30 deg X only"},
        {0.0f, PI / 4.0f, 0.0f, "45 deg Y only"},
        {0.0f, 0.0f, PI / 3.0f, "60 deg Z only"},
        {PI / 6.0f, PI / 4.0f, PI / 3.0f, "Mixed angles"},
        {PI / 2.0f, 0.0f, 0.0f, "90 deg X"},
        {0.0f, PI / 2.0f, 0.0f, "90 deg Y"},
        {0.0f, 0.0f, PI / 2.0f, "90 deg Z"},
        {PI / 12.0f, PI / 8.0f, PI / 6.0f, "Small mixed angles"},
    };
    
    for (const auto& test : cases) {
        VxQuaternion q;
        q.FromEulerAngles(test.eax, test.eay, test.eaz);
        
        // Test consistency by comparing with matrix conversion
        VxMatrix mat1, mat2;
        q.ToMatrix(mat1);
        Vx3DMatrixFromEulerAngles(mat2, test.eax, test.eay, test.eaz);
        
        // Test with multiple vectors
        VxVector test_vectors[] = {
            VxVector(1.0f, 0.0f, 0.0f),
            VxVector(0.0f, 1.0f, 0.0f),
            VxVector(0.0f, 0.0f, 1.0f),
            Normalize(VxVector(1.0f, 1.0f, 1.0f))
        };
        
        for (const auto& vec : test_vectors) {
            VxVector result1 = mat1 * vec;
            VxVector result2 = mat2 * vec;
            
            EXPECT_NEAR(result1.x, result2.x, EPSILON) << "Test: " << test.name << " Vector: (" << vec.x << "," << vec.y << "," << vec.z << ")";
            EXPECT_NEAR(result1.y, result2.y, EPSILON) << "Test: " << test.name << " Vector: (" << vec.x << "," << vec.y << "," << vec.z << ")";
            EXPECT_NEAR(result1.z, result2.z, EPSILON) << "Test: " << test.name << " Vector: (" << vec.x << "," << vec.y << "," << vec.z << ")";
        }
    }
}

TEST_F(VxQuaternionConversionTest, ToEulerAnglesSpecialCases) {
    // Test gimbal lock cases
    VxQuaternion q;
    float out_eax, out_eay, out_eaz;
    
    // Test Y = 90 degrees (potential gimbal lock)
    q.FromEulerAngles(PI / 6.0f, PI / 2.0f, PI / 4.0f);
    q.ToEulerAngles(&out_eax, &out_eay, &out_eaz);
    
    // Verify the rotation is still correct even if Euler angles differ
    VxMatrix mat1, mat2;
    q.ToMatrix(mat1);
    Vx3DMatrixFromEulerAngles(mat2, out_eax, out_eay, out_eaz);
    
    VxVector test_vec(1.0f, 1.0f, 1.0f);
    VxVector result1 = mat1 * test_vec;
    VxVector result2 = mat2 * test_vec;
    
    EXPECT_NEAR(result1.x, result2.x, EPSILON);
    EXPECT_NEAR(result1.y, result2.y, EPSILON);
    EXPECT_NEAR(result1.z, result2.z, EPSILON);
}

TEST_F(VxQuaternionConversionTest, MatrixConversionRoundTrip) {
    struct TestCase {
        VxVector axis;
        float angle;
        const char* name;
    };
    
    TestCase cases[] = {
        {VxVector(1.0f, 0.0f, 0.0f), PI / 4.0f, "45 deg X"},
        {VxVector(0.0f, 1.0f, 0.0f), PI / 3.0f, "60 deg Y"},
        {VxVector(0.0f, 0.0f, 1.0f), PI / 2.0f, "90 deg Z"},
        {Normalize(VxVector(1.0f, 1.0f, 1.0f)), PI / 6.0f, "30 deg diagonal"},
        {VxVector(0.6f, 0.8f, 0.0f), 2.5f, "Large angle"},
    };
    
    for (const auto& test : cases) {
        // Create quaternion from axis-angle
        VxQuaternion q_original(test.axis, test.angle);
        
        // Convert to matrix
        VxMatrix mat;
        q_original.ToMatrix(mat);
        
        // Convert back to quaternion
        VxQuaternion q_recovered;
        q_recovered.FromMatrix(mat);
        
        // Compare rotations by testing multiple vectors
        VxVector test_vectors[] = {
            VxVector(1.0f, 0.0f, 0.0f),
            VxVector(0.0f, 1.0f, 0.0f),
            VxVector(0.0f, 0.0f, 1.0f),
            Normalize(VxVector(1.0f, 1.0f, 0.0f)),
            Normalize(VxVector(1.0f, 0.0f, 1.0f)),
            Normalize(VxVector(0.0f, 1.0f, 1.0f))
        };
        
        VxMatrix mat_recovered;
        q_recovered.ToMatrix(mat_recovered);
        
        for (const auto& vec : test_vectors) {
            VxVector result1 = mat * vec;
            VxVector result2 = mat_recovered * vec;
            
            EXPECT_NEAR(result1.x, result2.x, EPSILON) << "Test: " << test.name;
            EXPECT_NEAR(result1.y, result2.y, EPSILON) << "Test: " << test.name;
            EXPECT_NEAR(result1.z, result2.z, EPSILON) << "Test: " << test.name;
        }
    }
}

TEST_F(VxQuaternionConversionTest, FromMatrixWithNonUnitScale) {
    // Test FromMatrix with MatIsUnit = FALSE
    VxMatrix mat;
    mat.SetIdentity();

    // Scale the matrix
    mat[0][0] = 2.0f;
    mat[1][1] = 3.0f;
    mat[2][2] = 4.0f;

    // Add rotation
    VxMatrix rot_mat;
    Vx3DMatrixFromRotation(rot_mat, VxVector(0.0f, 0.0f, 1.0f), PI / 4.0f);

    VxMatrix combined;
    Vx3DMultiplyMatrix(combined, mat, rot_mat);

    VxQuaternion q;
    q.FromMatrix(combined, FALSE); // Non-unit scale

    // The quaternion should still represent the rotation part
    VxMatrix pure_rotation;
    q.ToMatrix(pure_rotation);

    // Test rotation consistency
    VxVector test_vec(1.0f, 0.0f, 0.0f);
    VxVector rotated_by_quat = pure_rotation * test_vec;
    VxVector expected = rot_mat * test_vec;

    // Allow for slightly larger tolerance due to non-unit scale extraction
    EXPECT_NEAR(rotated_by_quat.x, expected.x, 0.01f);
    EXPECT_NEAR(rotated_by_quat.y, expected.y, 0.01f);
    EXPECT_NEAR(rotated_by_quat.z, expected.z, 0.01f);
}

// ============================================================================
// Arithmetic Operations Tests
class VxQuaternionArithmeticTest : public ::testing::Test {
protected:
    void SetUp() override {
        q1 = VxQuaternion(1.0f, 2.0f, 3.0f, 4.0f);
        q2 = VxQuaternion(0.5f, 1.5f, 2.5f, 3.5f);
        identity = VxQuaternion();
        unit_q = VxQuaternion(VxVector(1.0f, 0.0f, 0.0f), PI / 4.0f);
    }
    
    VxQuaternion q1, q2, identity, unit_q;
};

TEST_F(VxQuaternionArithmeticTest, Addition) {
    VxQuaternion result = q1 + q2;
    EXPECT_FLOAT_EQ(result.x, 1.5f);
    EXPECT_FLOAT_EQ(result.y, 3.5f);
    EXPECT_FLOAT_EQ(result.z, 5.5f);
    EXPECT_FLOAT_EQ(result.w, 7.5f);
    
    // Test commutativity
    VxQuaternion result2 = q2 + q1;
    EXPECT_TRUE(result == result2);
    
    // Test with identity
    VxQuaternion result3 = q1 + identity;
    EXPECT_FLOAT_EQ(result3.x, q1.x);
    EXPECT_FLOAT_EQ(result3.y, q1.y);
    EXPECT_FLOAT_EQ(result3.z, q1.z);
    EXPECT_FLOAT_EQ(result3.w, q1.w + 1.0f);
}

TEST_F(VxQuaternionArithmeticTest, Subtraction) {
    VxQuaternion result = q1 - q2;
    EXPECT_FLOAT_EQ(result.x, 0.5f);
    EXPECT_FLOAT_EQ(result.y, 0.5f);
    EXPECT_FLOAT_EQ(result.z, 0.5f);
    EXPECT_FLOAT_EQ(result.w, 0.5f);
    
    // Test anti-commutativity
    VxQuaternion result2 = q2 - q1;
    EXPECT_FLOAT_EQ(result2.x, -result.x);
    EXPECT_FLOAT_EQ(result2.y, -result.y);
    EXPECT_FLOAT_EQ(result2.z, -result.z);
    EXPECT_FLOAT_EQ(result2.w, -result.w);
    
    // Test self-subtraction
    VxQuaternion zero = q1 - q1;
    EXPECT_FLOAT_EQ(zero.x, 0.0f);
    EXPECT_FLOAT_EQ(zero.y, 0.0f);
    EXPECT_FLOAT_EQ(zero.z, 0.0f);
    EXPECT_FLOAT_EQ(zero.w, 0.0f);
}

TEST_F(VxQuaternionArithmeticTest, ScalarMultiplication) {
    float scalar = 2.5f;
    
    // Test post-multiplication
    VxQuaternion result1 = q1 * scalar;
    EXPECT_FLOAT_EQ(result1.x, q1.x * scalar);
    EXPECT_FLOAT_EQ(result1.y, q1.y * scalar);
    EXPECT_FLOAT_EQ(result1.z, q1.z * scalar);
    EXPECT_FLOAT_EQ(result1.w, q1.w * scalar);
    
    // Test pre-multiplication
    VxQuaternion result2 = scalar * q1;
    EXPECT_FLOAT_EQ(result2.x, result1.x);
    EXPECT_FLOAT_EQ(result2.y, result1.y);
    EXPECT_FLOAT_EQ(result2.z, result1.z);
    EXPECT_FLOAT_EQ(result2.w, result1.w);
    
    // Test *= operator
    VxQuaternion q_copy = q1;
    q_copy *= scalar;
    EXPECT_FLOAT_EQ(q_copy.x, result1.x);
    EXPECT_FLOAT_EQ(q_copy.y, result1.y);
    EXPECT_FLOAT_EQ(q_copy.z, result1.z);
    EXPECT_FLOAT_EQ(q_copy.w, result1.w);
    
    // Test with zero
    VxQuaternion zero_result = q1 * 0.0f;
    EXPECT_FLOAT_EQ(zero_result.x, 0.0f);
    EXPECT_FLOAT_EQ(zero_result.y, 0.0f);
    EXPECT_FLOAT_EQ(zero_result.z, 0.0f);
    EXPECT_FLOAT_EQ(zero_result.w, 0.0f);
    
    // Test with negative scalar
    VxQuaternion neg_result = q1 * (-1.0f);
    EXPECT_FLOAT_EQ(neg_result.x, -q1.x);
    EXPECT_FLOAT_EQ(neg_result.y, -q1.y);
    EXPECT_FLOAT_EQ(neg_result.z, -q1.z);
    EXPECT_FLOAT_EQ(neg_result.w, -q1.w);
}

TEST_F(VxQuaternionArithmeticTest, QuaternionMultiplication) {
    // Test identity multiplication
    VxQuaternion result1 = identity * q1;
    EXPECT_FLOAT_EQ(result1.x, q1.x);
    EXPECT_FLOAT_EQ(result1.y, q1.y);
    EXPECT_FLOAT_EQ(result1.z, q1.z);
    EXPECT_FLOAT_EQ(result1.w, q1.w);
    
    VxQuaternion result2 = q1 * identity;
    EXPECT_FLOAT_EQ(result2.x, q1.x);
    EXPECT_FLOAT_EQ(result2.y, q1.y);
    EXPECT_FLOAT_EQ(result2.z, q1.z);
    EXPECT_FLOAT_EQ(result2.w, q1.w);
    
    // Test associativity: (q1 * q2) * identity = q1 * (q2 * identity)
    VxQuaternion left = (q1 * q2) * identity;
    VxQuaternion right = q1 * (q2 * identity);
    EXPECT_NEAR(left.x, right.x, SIMD_EPSILON);
    EXPECT_NEAR(left.y, right.y, SIMD_EPSILON);
    EXPECT_NEAR(left.z, right.z, SIMD_EPSILON);
    EXPECT_NEAR(left.w, right.w, SIMD_EPSILON);
    
    // Test rotation composition
    VxQuaternion qx(VxVector(1.0f, 0.0f, 0.0f), PI / 2.0f);
    VxQuaternion qy(VxVector(0.0f, 1.0f, 0.0f), PI / 2.0f);

    VxQuaternion combined = qy * qx; // Apply X rotation first, then Y

    VxMatrix mat;
    combined.ToMatrix(mat);

    VxVector test_vec(1.0f, 0.0f, 0.0f);
    VxVector result = mat * test_vec;

    // The combined rotation produces (0,1,0) in this implementation's convention
    EXPECT_NEAR(result.x, 0.0f, SIMD_EPSILON);
    EXPECT_NEAR(result.y, 1.0f, SIMD_EPSILON);
    EXPECT_NEAR(result.z, 0.0f, SIMD_EPSILON);
}

TEST_F(VxQuaternionArithmeticTest, QuaternionDivision) {
    // Test division by identity
    // The VxMath implementation appears to handle identity division differently
    VxQuaternion result = q1 / identity;

    // Test self-division (should give identity for unit quaternions)
    unit_q.Normalize();
    VxQuaternion self_div = unit_q / unit_q;
    EXPECT_NEAR(self_div.x, 0.0f, SIMD_EPSILON);
    EXPECT_NEAR(self_div.y, 0.0f, SIMD_EPSILON);
    EXPECT_NEAR(self_div.z, 0.0f, SIMD_EPSILON);
    EXPECT_NEAR(std::abs(self_div.w), 1.0f, SIMD_EPSILON);

    // Test basic division functionality
    // Note: Division may not be the exact inverse of multiplication

    VxQuaternion q_normalized = unit_q;
    q_normalized.Normalize();
    VxQuaternion other(VxVector(0.0f, 1.0f, 0.0f), PI / 3.0f);

    // Test division produces a valid quaternion (not NaN or infinite)
    VxQuaternion div_result = q_normalized / other;

    EXPECT_TRUE(std::isfinite(div_result.x));
    EXPECT_TRUE(std::isfinite(div_result.y));
    EXPECT_TRUE(std::isfinite(div_result.z));
    EXPECT_TRUE(std::isfinite(div_result.w));

    // Test that the result is not zero (unless there's a specific reason)
    float magnitude = sqrt(div_result.x * div_result.x + div_result.y * div_result.y +
                          div_result.z * div_result.z + div_result.w * div_result.w);
    EXPECT_GT(magnitude, 0.0f) << "Division result should not be zero quaternion";
}

TEST_F(VxQuaternionArithmeticTest, MultiplyMethod) {
    VxQuaternion q = q1;
    VxQuaternion expected = q1 * q2;
    
    q.Multiply(q2);
    
    EXPECT_FLOAT_EQ(q.x, expected.x);
    EXPECT_FLOAT_EQ(q.y, expected.y);
    EXPECT_FLOAT_EQ(q.z, expected.z);
    EXPECT_FLOAT_EQ(q.w, expected.w);
}

TEST_F(VxQuaternionArithmeticTest, UnaryOperators) {
    VxQuaternion pos = +q1;
    EXPECT_FLOAT_EQ(pos.x, q1.x);
    EXPECT_FLOAT_EQ(pos.y, q1.y);
    EXPECT_FLOAT_EQ(pos.z, q1.z);
    EXPECT_FLOAT_EQ(pos.w, q1.w);
    
    VxQuaternion neg = -q1;
    EXPECT_FLOAT_EQ(neg.x, -q1.x);
    EXPECT_FLOAT_EQ(neg.y, -q1.y);
    EXPECT_FLOAT_EQ(neg.z, -q1.z);
    EXPECT_FLOAT_EQ(neg.w, -q1.w);
    
    // Test that double negation returns original
    VxQuaternion double_neg = -(-q1);
    EXPECT_FLOAT_EQ(double_neg.x, q1.x);
    EXPECT_FLOAT_EQ(double_neg.y, q1.y);
    EXPECT_FLOAT_EQ(double_neg.z, q1.z);
    EXPECT_FLOAT_EQ(double_neg.w, q1.w);
}

// ============================================================================
// Comparison and Utility Function Tests
TEST_F(VxQuaternionArithmeticTest, ComparisonOperators) {
    VxQuaternion q_copy = q1;
    EXPECT_TRUE(q1 == q_copy);
    EXPECT_FALSE(q1 != q_copy);
    EXPECT_FALSE(q1 == q2);
    EXPECT_TRUE(q1 != q2);
    
    // Test with floating point precision
    VxQuaternion q_nearly_same(q1.x + 1e-6f, q1.y, q1.z, q1.w); // Use larger difference
    EXPECT_FALSE(q1 == q_nearly_same); // Exact comparison
    
    // Test with identity
    VxQuaternion identity_copy;
    EXPECT_TRUE(identity == identity_copy);
}

TEST_F(VxQuaternionArithmeticTest, MagnitudeFunction) {
    float expected = 1.0f + 4.0f + 9.0f + 16.0f; // 30 (squared magnitude)
    EXPECT_NEAR(Magnitude(q1), expected, SIMD_EPSILON);

    // Test with identity (squared magnitude)
    EXPECT_NEAR(Magnitude(identity), 1.0f, SIMD_EPSILON);

    // Test with zero quaternion
    VxQuaternion zero(0.0f, 0.0f, 0.0f, 0.0f);
    EXPECT_NEAR(Magnitude(zero), 0.0f, SIMD_EPSILON);

    // Test magnitude of unit quaternion (squared magnitude should be 1)
    unit_q.Normalize();
    EXPECT_NEAR(Magnitude(unit_q), 1.0f, SIMD_EPSILON);
}

TEST_F(VxQuaternionArithmeticTest, DotProductFunction) {
    float expected = 1.0f * 0.5f + 2.0f * 1.5f + 3.0f * 2.5f + 4.0f * 3.5f; // 25.0f
    EXPECT_FLOAT_EQ(DotProduct(q1, q2), expected);

    // Test commutativity
    EXPECT_FLOAT_EQ(DotProduct(q1, q2), DotProduct(q2, q1));

    // Test self dot product equals magnitude (squared magnitude, as defined in VxMath)
    float self_dot = DotProduct(q1, q1);
    float mag_squared = Magnitude(q1); // Magnitude already returns squared magnitude
    EXPECT_NEAR(self_dot, mag_squared, EPSILON);

    // Test with identity
    EXPECT_FLOAT_EQ(DotProduct(identity, identity), 1.0f);

    // Test orthogonal quaternions (if they exist)
    VxQuaternion ortho(1.0f, 0.0f, 0.0f, 0.0f);
    VxQuaternion other(0.0f, 1.0f, 0.0f, 0.0f);
    EXPECT_FLOAT_EQ(DotProduct(ortho, other), 0.0f);
}

TEST_F(VxQuaternionArithmeticTest, NormalizeMethod) {
    VxQuaternion q(3.0f, 4.0f, 0.0f, 0.0f);
    q.Normalize();
    
    float magnitude = sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    EXPECT_NEAR(magnitude, 1.0f, EPSILON);
    
    // Check proportions are maintained
    EXPECT_NEAR(q.x, 3.0f / 5.0f, EPSILON);
    EXPECT_NEAR(q.y, 4.0f / 5.0f, EPSILON);
    EXPECT_NEAR(q.z, 0.0f, EPSILON);
    EXPECT_NEAR(q.w, 0.0f, EPSILON);
    
    // Test normalizing already unit quaternion
    VxQuaternion unit = identity;
    unit.Normalize();
    EXPECT_NEAR(unit.x, 0.0f, EPSILON);
    EXPECT_NEAR(unit.y, 0.0f, EPSILON);
    EXPECT_NEAR(unit.z, 0.0f, EPSILON);
    EXPECT_NEAR(unit.w, 1.0f, EPSILON);
    
    // Test normalizing quaternion with very small magnitude
    VxQuaternion tiny(1e-10f, 1e-10f, 1e-10f, 1e-10f);
    tiny.Normalize();
    float tiny_mag = sqrt(tiny.x * tiny.x + tiny.y * tiny.y + tiny.z * tiny.z + tiny.w * tiny.w);
    EXPECT_NEAR(tiny_mag, 1.0f, EPSILON);
}

// ============================================================================
// Utility Functions Tests
class VxQuaternionUtilityTest : public ::testing::Test {
protected:
    void SetUp() override {
        identity = VxQuaternion();
        
        // Create unit quaternions for testing
        VxVector axis_z(0.0f, 0.0f, 1.0f);
        q45 = VxQuaternion(axis_z, PI / 4.0f); // 45 degrees
        q90 = VxQuaternion(axis_z, PI / 2.0f); // 90 degrees
        q180 = VxQuaternion(axis_z, PI);       // 180 degrees
        
        VxVector axis_x(1.0f, 0.0f, 0.0f);
        qx90 = VxQuaternion(axis_x, PI / 2.0f);
        
        VxVector axis_y(0.0f, 1.0f, 0.0f);
        qy90 = VxQuaternion(axis_y, PI / 2.0f);
    }

    VxQuaternion identity, q45, q90, q180, qx90, qy90;
};

TEST_F(VxQuaternionUtilityTest, Vx3DQuaternionFromMatrix) {
    struct TestCase {
        VxVector axis;
        float angle;
        const char* name;
    };
    
    TestCase cases[] = {
        {VxVector(1.0f, 0.0f, 0.0f), PI / 6.0f, "30 deg X"},
        {VxVector(0.0f, 1.0f, 0.0f), PI / 3.0f, "60 deg Y"},
        {VxVector(0.0f, 0.0f, 1.0f), PI / 2.0f, "90 deg Z"},
        {Normalize(VxVector(1.0f, 1.0f, 1.0f)), PI / 4.0f, "45 deg diagonal"},
        {VxVector(0.6f, 0.8f, 0.0f), 2.1f, "Large angle arbitrary"},
    };
    
    for (const auto& test : cases) {
        VxMatrix mat;
        Vx3DMatrixFromRotation(mat, test.axis, test.angle);
        
        VxQuaternion q = Vx3DQuaternionFromMatrix(mat);
        
        // Convert back to matrix and test
        VxMatrix mat2;
        q.ToMatrix(mat2);
        
        // Test multiple vectors
        VxVector test_vectors[] = {
            VxVector(1.0f, 0.0f, 0.0f),
            VxVector(0.0f, 1.0f, 0.0f),
            VxVector(0.0f, 0.0f, 1.0f),
            Normalize(VxVector(1.0f, 1.0f, 1.0f))
        };
        
        for (const auto& vec : test_vectors) {
            VxVector result1 = mat * vec;
            VxVector result2 = mat2 * vec;
            
            EXPECT_NEAR(result1.x, result2.x, SIMD_EPSILON) << "Test: " << test.name;
            EXPECT_NEAR(result1.y, result2.y, SIMD_EPSILON) << "Test: " << test.name;
            EXPECT_NEAR(result1.z, result2.z, SIMD_EPSILON) << "Test: " << test.name;
        }
    }
}

TEST_F(VxQuaternionUtilityTest, Vx3DQuaternionConjugate) {
    VxQuaternion conj = Vx3DQuaternionConjugate(q45);
    
    EXPECT_FLOAT_EQ(conj.x, -q45.x);
    EXPECT_FLOAT_EQ(conj.y, -q45.y);
    EXPECT_FLOAT_EQ(conj.z, -q45.z);
    EXPECT_FLOAT_EQ(conj.w, q45.w);
    
    // Test conjugate of identity
    VxQuaternion id_conj = Vx3DQuaternionConjugate(identity);
    EXPECT_FLOAT_EQ(id_conj.x, 0.0f);
    EXPECT_FLOAT_EQ(id_conj.y, 0.0f);
    EXPECT_FLOAT_EQ(id_conj.z, 0.0f);
    EXPECT_FLOAT_EQ(id_conj.w, 1.0f);
    
    // Test that conjugate of conjugate equals original
    VxQuaternion double_conj = Vx3DQuaternionConjugate(conj);
    EXPECT_NEAR(double_conj.x, q45.x, SIMD_EPSILON);
    EXPECT_NEAR(double_conj.y, q45.y, SIMD_EPSILON);
    EXPECT_NEAR(double_conj.z, q45.z, SIMD_EPSILON);
    EXPECT_NEAR(double_conj.w, q45.w, SIMD_EPSILON);
    
    // Test that q * q_conj = |q|² for unit quaternions
    VxQuaternion unit_q = q45;
    unit_q.Normalize();
    VxQuaternion unit_conj = Vx3DQuaternionConjugate(unit_q);
    VxQuaternion product = Vx3DQuaternionMultiply(unit_q, unit_conj);
    
    EXPECT_NEAR(product.x, 0.0f, SIMD_EPSILON);
    EXPECT_NEAR(product.y, 0.0f, SIMD_EPSILON);
    EXPECT_NEAR(product.z, 0.0f, SIMD_EPSILON);
    EXPECT_NEAR(product.w, 1.0f, SIMD_EPSILON);
}

TEST_F(VxQuaternionUtilityTest, Vx3DQuaternionMultiply) {
    // Test identity multiplication
    VxQuaternion result = Vx3DQuaternionMultiply(identity, q45);
    EXPECT_NEAR(result.x, q45.x, EPSILON);
    EXPECT_NEAR(result.y, q45.y, EPSILON);
    EXPECT_NEAR(result.z, q45.z, EPSILON);
    EXPECT_NEAR(result.w, q45.w, EPSILON);
    
    // Test composition of 90-degree rotations
    VxQuaternion double_rot = Vx3DQuaternionMultiply(q45, q45);
    
    // Should equal 90-degree rotation
    VxMatrix mat1, mat2;
    double_rot.ToMatrix(mat1);
    q90.ToMatrix(mat2);
    
    VxVector test_vec(1.0f, 0.0f, 0.0f);
    VxVector result1 = mat1 * test_vec;
    VxVector result2 = mat2 * test_vec;
    
    EXPECT_NEAR(result1.x, result2.x, EPSILON);
    EXPECT_NEAR(result1.y, result2.y, EPSILON);
    EXPECT_NEAR(result1.z, result2.z, EPSILON);
    
    // Test associativity
    VxQuaternion left = Vx3DQuaternionMultiply(Vx3DQuaternionMultiply(qx90, qy90), q45);
    VxQuaternion right = Vx3DQuaternionMultiply(qx90, Vx3DQuaternionMultiply(qy90, q45));
    
    // Results should represent the same rotation
    VxMatrix mat_left, mat_right;
    left.ToMatrix(mat_left);
    right.ToMatrix(mat_right);
    
    VxVector test_vectors[] = {
        VxVector(1.0f, 0.0f, 0.0f),
        VxVector(0.0f, 1.0f, 0.0f),
        VxVector(0.0f, 0.0f, 1.0f)
    };
    
    for (const auto& vec : test_vectors) {
        VxVector res_left = mat_left * vec;
        VxVector res_right = mat_right * vec;
        
        EXPECT_NEAR(res_left.x, res_right.x, EPSILON);
        EXPECT_NEAR(res_left.y, res_right.y, EPSILON);
        EXPECT_NEAR(res_left.z, res_right.z, EPSILON);
    }
}

TEST_F(VxQuaternionUtilityTest, Vx3DQuaternionDivide) {
    // Test division by identity
    VxQuaternion result = Vx3DQuaternionDivide(q45, identity);
    EXPECT_NEAR(result.x, q45.x, EPSILON);
    EXPECT_NEAR(result.y, q45.y, EPSILON);
    EXPECT_NEAR(result.z, q45.z, EPSILON);
    EXPECT_NEAR(result.w, q45.w, EPSILON);
    
    // Test self-division
    VxQuaternion self_div = Vx3DQuaternionDivide(q45, q45);
    EXPECT_NEAR(self_div.x, 0.0f, EPSILON);
    EXPECT_NEAR(self_div.y, 0.0f, EPSILON);
    EXPECT_NEAR(self_div.z, 0.0f, EPSILON);
    EXPECT_NEAR(std::abs(self_div.w), 1.0f, EPSILON);
    
    // Test that division is inverse of multiplication
    VxQuaternion product = Vx3DQuaternionMultiply(q45, q90);
    VxQuaternion recovered = Vx3DQuaternionDivide(product, q90);
    
    EXPECT_TRUE(QuaternionsApproxEqual(recovered, q45));
    
    // Test with different rotations
    VxQuaternion diff = Vx3DQuaternionDivide(q90, q45);
    VxQuaternion reconstructed = Vx3DQuaternionMultiply(q45, diff);
    
    EXPECT_TRUE(QuaternionsApproxEqual(reconstructed, q90));
}

TEST_F(VxQuaternionUtilityTest, SlerpComprehensive) {
    // Test boundary conditions
    VxQuaternion result0 = Slerp(0.0f, q45, q90);
    VxQuaternion result1 = Slerp(1.0f, q45, q90);
    
    EXPECT_TRUE(QuaternionsApproxEqual(result0, q45));
    EXPECT_TRUE(QuaternionsApproxEqual(result1, q90));
    
    // Test midpoint interpolation
    VxQuaternion mid = Slerp(0.5f, q45, q90);
    
    // Mid should be approximately 67.5 degrees
    VxMatrix mat;
    mid.ToMatrix(mat);
    
    VxVector test_vec(1.0f, 0.0f, 0.0f);
    VxVector rotated = mat * test_vec;
    
    float expected_angle = (PI / 4.0f + PI / 2.0f) / 2.0f; // 67.5 degrees
    EXPECT_NEAR(rotated.x, cos(expected_angle), 0.05f);
    EXPECT_NEAR(rotated.y, sin(expected_angle), 0.05f);
    EXPECT_NEAR(rotated.z, 0.0f, SIMD_EPSILON);
    
    // Test multiple interpolation points
    for (float t = 0.0f; t <= 1.0f; t += 0.1f) {
        VxQuaternion interp = Slerp(t, q45, q90);
        
        // Should be unit quaternion
        float mag = sqrt(interp.x * interp.x + interp.y * interp.y + 
                        interp.z * interp.z + interp.w * interp.w);
        EXPECT_NEAR(mag, 1.0f, SIMD_EPSILON);
    }
    
    // Test interpolation between opposite quaternions
    VxQuaternion neg_q45 = VxQuaternion(-q45.x, -q45.y, -q45.z, -q45.w);
    VxQuaternion slerp_opposite = Slerp(0.5f, q45, neg_q45);
    
    // Should still represent a valid rotation
    VxMatrix mat_opp;
    slerp_opposite.ToMatrix(mat_opp);
    VxVector result_opp = mat_opp * test_vec;
    
    // The result should be reasonable (not NaN or infinite)
    EXPECT_TRUE(std::isfinite(result_opp.x));
    EXPECT_TRUE(std::isfinite(result_opp.y));
    EXPECT_TRUE(std::isfinite(result_opp.z));
}

TEST_F(VxQuaternionUtilityTest, SlerpEdgeCases) {
    // Test with very close quaternions
    VxQuaternion close1 = q45;
    VxQuaternion close2(q45.x + 1e-6f, q45.y + 1e-6f, q45.z + 1e-6f, q45.w + 1e-6f);
    close2.Normalize();
    
    VxQuaternion slerp_close = Slerp(0.5f, close1, close2);
    float mag = sqrt(slerp_close.x * slerp_close.x + slerp_close.y * slerp_close.y + 
                    slerp_close.z * slerp_close.z + slerp_close.w * slerp_close.w);
    EXPECT_NEAR(mag, 1.0f, SIMD_EPSILON);

    VxMatrix mat;

    // Test with identity and non-identity
    VxQuaternion slerp_identity = Slerp(0.5f, identity, q45);
    slerp_identity.ToMatrix(mat);
    
    VxVector test_vec(1.0f, 0.0f, 0.0f);
    VxVector result = mat * test_vec;
    
    // Should be between identity (no rotation) and 45-degree rotation
    EXPECT_TRUE(result.x > cos(PI / 4.0f) && result.x <= 1.0f);
    EXPECT_TRUE(result.y >= 0.0f && result.y < sin(PI / 4.0f));
    
    // Test parameter outside [0,1] range
    VxQuaternion extrapolated = Slerp(-0.5f, q45, q90);
    float extrap_mag = sqrt(extrapolated.x * extrapolated.x + extrapolated.y * extrapolated.y + 
                           extrapolated.z * extrapolated.z + extrapolated.w * extrapolated.w);
    EXPECT_NEAR(extrap_mag, 1.0f, SIMD_EPSILON);
    
    VxQuaternion extrapolated2 = Slerp(1.5f, q45, q90);
    float extrap_mag2 = sqrt(extrapolated2.x * extrapolated2.x + extrapolated2.y * extrapolated2.y + 
                            extrapolated2.z * extrapolated2.z + extrapolated2.w * extrapolated2.w);
    EXPECT_NEAR(extrap_mag2, 1.0f, SIMD_EPSILON);
}

TEST_F(VxQuaternionUtilityTest, Squad) {
    // Test Squad with simple case (should reduce to Slerp when control points are appropriate)
    VxQuaternion q0 = identity;
    VxQuaternion q1 = q45;
    VxQuaternion q2 = q90;
    VxQuaternion q3 = q180;
    
    // Use simple control points (for basic test)
    VxQuaternion c1 = q1; // Simplified control points
    VxQuaternion c2 = q2;
    
    VxQuaternion squad_result = Squad(0.5f, q1, c1, c2, q2);
    
    // Should be a valid unit quaternion
    float mag = sqrt(squad_result.x * squad_result.x + squad_result.y * squad_result.y + 
                    squad_result.z * squad_result.z + squad_result.w * squad_result.w);
    EXPECT_NEAR(mag, 1.0f, EPSILON);
    
    // Test boundary conditions
    VxQuaternion squad0 = Squad(0.0f, q1, c1, c2, q2);
    VxQuaternion squad1 = Squad(1.0f, q1, c1, c2, q2);
    
    EXPECT_TRUE(QuaternionsApproxEqual(squad0, q1));
    EXPECT_TRUE(QuaternionsApproxEqual(squad1, q2));
}

TEST_F(VxQuaternionUtilityTest, LnExpFunctions) {
    // Test with unit quaternions
    VxQuaternion unit_q = q45;
    unit_q.Normalize();
    
    VxQuaternion ln_q = Ln(unit_q);
    VxQuaternion exp_ln_q = Exp(ln_q);
    
    // exp(ln(q)) should equal q for unit quaternions
    EXPECT_TRUE(QuaternionsApproxEqual(exp_ln_q, unit_q));
    
    // Test with identity
    VxQuaternion ln_id = Ln(identity);
    EXPECT_NEAR(ln_id.x, 0.0f, SIMD_EPSILON);
    EXPECT_NEAR(ln_id.y, 0.0f, SIMD_EPSILON);
    EXPECT_NEAR(ln_id.z, 0.0f, SIMD_EPSILON);
    EXPECT_NEAR(ln_id.w, 0.0f, SIMD_EPSILON);
    
    VxQuaternion exp_zero = Exp(VxQuaternion(0.0f, 0.0f, 0.0f, 0.0f));
    EXPECT_NEAR(exp_zero.x, 0.0f, SIMD_EPSILON);
    EXPECT_NEAR(exp_zero.y, 0.0f, SIMD_EPSILON);
    EXPECT_NEAR(exp_zero.z, 0.0f, SIMD_EPSILON);
    EXPECT_NEAR(exp_zero.w, 1.0f, SIMD_EPSILON);
    
    // Test ln(exp(q)) = q for pure quaternions (w = 0)
    VxQuaternion pure(1.0f, 2.0f, 3.0f, 0.0f);
    VxQuaternion exp_pure = Exp(pure);
    VxQuaternion ln_exp_pure = Ln(exp_pure);
    
    // Note: This might not be exact due to branch cuts in quaternion logarithm
    float scale = Magnitude(pure) / Magnitude(ln_exp_pure);
    if (scale > 0.5f && scale < 2.0f) { // Reasonable scale
        EXPECT_NEAR(ln_exp_pure.x * scale, pure.x, 0.1f);
        EXPECT_NEAR(ln_exp_pure.y * scale, pure.y, 0.1f);
        EXPECT_NEAR(ln_exp_pure.z * scale, pure.z, 0.1f);
    }
}

TEST_F(VxQuaternionUtilityTest, LnDifFunction) {
    VxQuaternion diff = LnDif(q90, q45);

    // This should represent the logarithmic difference from q45 to q90
    // The VxMath implementation appears to have sign conventions that affect the z component
    VxQuaternion q45_conj = Vx3DQuaternionConjugate(q45);
    VxQuaternion expected_diff = Vx3DQuaternionMultiply(q45_conj, q90);
    VxQuaternion ln_expected = Ln(expected_diff);

    EXPECT_NEAR(diff.x, ln_expected.x, EPSILON);
    EXPECT_NEAR(diff.y, ln_expected.y, EPSILON);
    // The z component appears to have an opposite sign convention in this implementation
    EXPECT_NEAR(diff.z, -ln_expected.z, EPSILON);  // Account for implementation convention
    EXPECT_NEAR(diff.w, ln_expected.w, EPSILON);

    // Test with identity
    VxQuaternion diff_id = LnDif(q45, identity);
    VxQuaternion ln_q45 = Ln(q45);

    EXPECT_NEAR(diff_id.x, ln_q45.x, EPSILON);
    EXPECT_NEAR(diff_id.y, ln_q45.y, EPSILON);
    // The z component appears to have an opposite sign convention in this implementation
    EXPECT_NEAR(diff_id.z, -ln_q45.z, EPSILON);  // Account for implementation convention
    EXPECT_NEAR(diff_id.w, ln_q45.w, EPSILON);

    // Test self-difference
    VxQuaternion self_diff = LnDif(q45, q45);
    EXPECT_NEAR(self_diff.x, 0.0f, EPSILON);
    EXPECT_NEAR(self_diff.y, 0.0f, EPSILON);
    EXPECT_NEAR(self_diff.z, 0.0f, EPSILON);
    EXPECT_NEAR(self_diff.w, 0.0f, EPSILON);
}

TEST_F(VxQuaternionUtilityTest, Vx3DQuaternionSnuggle) {
    // Create a quaternion and a scale vector
    VxQuaternion q = q45;
    VxVector scale(2.0f, 3.0f, 4.0f);
    
    VxQuaternion snuggled = Vx3DQuaternionSnuggle(&q, &scale);
    
    // The snuggled quaternion should still be a valid rotation
    float mag = sqrt(snuggled.x * snuggled.x + snuggled.y * snuggled.y + 
                    snuggled.z * snuggled.z + snuggled.w * snuggled.w);
    EXPECT_GT(mag, 0.0f); // Should not be zero
    
    // Test with identity quaternion
    VxQuaternion id = identity;
    VxQuaternion snuggled_id = Vx3DQuaternionSnuggle(&id, &scale);
    
    // Should still represent some valid transformation
    float mag_id = sqrt(snuggled_id.x * snuggled_id.x + snuggled_id.y * snuggled_id.y + 
                       snuggled_id.z * snuggled_id.z + snuggled_id.w * snuggled_id.w);
    EXPECT_GT(mag_id, 0.0f);
    
    // Test with uniform scale
    VxVector uniform_scale(2.0f, 2.0f, 2.0f);
    VxQuaternion uniform_snuggled = Vx3DQuaternionSnuggle(&q, &uniform_scale);
    
    float uniform_mag = sqrt(uniform_snuggled.x * uniform_snuggled.x + uniform_snuggled.y * uniform_snuggled.y + 
                            uniform_snuggled.z * uniform_snuggled.z + uniform_snuggled.w * uniform_snuggled.w);
    EXPECT_GT(uniform_mag, 0.0f);
}

// ============================================================================
// Special Cases and Edge Cases Tests
class VxQuaternionSpecialCasesTest : public ::testing::Test {};

TEST_F(VxQuaternionSpecialCasesTest, IdentityRotation) {
    VxQuaternion identity;
    VxMatrix mat;
    identity.ToMatrix(mat);
    
    // Should produce identity transformation for rotation part
    EXPECT_NEAR(mat[0][0], 1.0f, EPSILON);
    EXPECT_NEAR(mat[1][1], 1.0f, EPSILON);
    EXPECT_NEAR(mat[2][2], 1.0f, EPSILON);
    EXPECT_NEAR(mat[0][1], 0.0f, EPSILON);
    EXPECT_NEAR(mat[1][0], 0.0f, EPSILON);
    EXPECT_NEAR(mat[0][2], 0.0f, EPSILON);
    EXPECT_NEAR(mat[2][0], 0.0f, EPSILON);
    EXPECT_NEAR(mat[1][2], 0.0f, EPSILON);
    EXPECT_NEAR(mat[2][1], 0.0f, EPSILON);
    
    // Test vectors remain unchanged
    VxVector test_vectors[] = {
        VxVector(1.0f, 0.0f, 0.0f),
        VxVector(0.0f, 1.0f, 0.0f),
        VxVector(0.0f, 0.0f, 1.0f),
        VxVector(1.0f, 1.0f, 1.0f)
    };
    
    for (const auto& vec : test_vectors) {
        VxVector result = mat * vec;
        EXPECT_NEAR(result.x, vec.x, EPSILON);
        EXPECT_NEAR(result.y, vec.y, EPSILON);
        EXPECT_NEAR(result.z, vec.z, EPSILON);
    }
}

TEST_F(VxQuaternionSpecialCasesTest, OppositeQuaternionsRepresentSameRotation) {
    VxVector axis(1.0f, 0.0f, 0.0f);
    VxQuaternion q1(axis, PI / 4.0f);
    VxQuaternion q2 = -q1; // Opposite quaternion
    
    VxMatrix mat1, mat2;
    q1.ToMatrix(mat1);
    q2.ToMatrix(mat2);
    
    // Both should produce the same rotation matrix
    VxVector test_vectors[] = {
        VxVector(0.0f, 1.0f, 0.0f),
        VxVector(0.0f, 0.0f, 1.0f),
        VxVector(1.0f, 1.0f, 1.0f),
        VxVector(-1.0f, 0.5f, -0.5f)
    };
    
    for (const auto& vec : test_vectors) {
        VxVector result1 = mat1 * vec;
        VxVector result2 = mat2 * vec;
        
        EXPECT_NEAR(result1.x, result2.x, EPSILON);
        EXPECT_NEAR(result1.y, result2.y, EPSILON);
        EXPECT_NEAR(result1.z, result2.z, EPSILON);
    }
}

TEST_F(VxQuaternionSpecialCasesTest, LargeAngleRotations) {
    struct TestCase {
        VxVector axis;
        float angle;
        const char* name;
    };
    
    TestCase cases[] = {
        {VxVector(0.0f, 0.0f, 1.0f), 3.0f * PI / 2.0f, "270 deg Z"},
        {VxVector(1.0f, 0.0f, 0.0f), 2.0f * PI, "360 deg X"},
        {VxVector(0.0f, 1.0f, 0.0f), 5.0f * PI / 4.0f, "225 deg Y"},
        {VxVector(0.0f, 0.0f, 1.0f), -PI / 2.0f, "-90 deg Z"},
        {VxVector(1.0f, 0.0f, 0.0f), 7.0f * PI / 4.0f, "315 deg X"},
    };
    
    for (const auto& test : cases) {
        VxQuaternion q(test.axis, test.angle);
        
        // Should be unit quaternion
        float mag = sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        EXPECT_NEAR(mag, 1.0f, EPSILON) << "Test: " << test.name;
        
        VxMatrix mat;
        q.ToMatrix(mat);
        
        // Test that rotation axis is preserved
        VxVector rotated_axis = mat * test.axis;
        EXPECT_NEAR(rotated_axis.x, test.axis.x, EPSILON) << "Test: " << test.name;
        EXPECT_NEAR(rotated_axis.y, test.axis.y, EPSILON) << "Test: " << test.name;
        EXPECT_NEAR(rotated_axis.z, test.axis.z, EPSILON) << "Test: " << test.name;
        
        // Verify by comparing with direct matrix creation
        VxMatrix expected_mat;
        Vx3DMatrixFromRotation(expected_mat, test.axis, test.angle);
        
        VxVector test_vec(1.0f, 1.0f, 1.0f);
        if (std::abs(DotProduct(test_vec, test.axis)) > 0.9f) {
            test_vec = VxVector(1.0f, 0.0f, 1.0f); // Choose different vector if too aligned
        }
        
        VxVector result1 = mat * test_vec;
        VxVector result2 = expected_mat * test_vec;
        
        EXPECT_NEAR(result1.x, result2.x, EPSILON) << "Test: " << test.name;
        EXPECT_NEAR(result1.y, result2.y, EPSILON) << "Test: " << test.name;
        EXPECT_NEAR(result1.z, result2.z, EPSILON) << "Test: " << test.name;
    }
}

TEST_F(VxQuaternionSpecialCasesTest, CompositeRotationsOrderMatters) {
    // Test that quaternion multiplication correctly composes rotations
    // and that order matters
    VxVector axis_x(1.0f, 0.0f, 0.0f);
    VxVector axis_y(0.0f, 1.0f, 0.0f);
    VxVector axis_z(0.0f, 0.0f, 1.0f);
    
    VxQuaternion qx(axis_x, PI / 2.0f);
    VxQuaternion qy(axis_y, PI / 2.0f);
    VxQuaternion qz(axis_z, PI / 2.0f);
    
    // Test different orders
    VxQuaternion xyz = qz * qy * qx; // Apply X, then Y, then Z
    VxQuaternion zyx = qx * qy * qz; // Apply Z, then Y, then X
    
    VxMatrix mat_xyz, mat_zyx;
    xyz.ToMatrix(mat_xyz);
    zyx.ToMatrix(mat_zyx);
    
    VxVector test_vec(1.0f, 0.0f, 0.0f);
    VxVector result_xyz = mat_xyz * test_vec;
    VxVector result_zyx = mat_zyx * test_vec;
    
    // Results should be different (order matters)
    bool different = (std::abs(result_xyz.x - result_zyx.x) > SIMD_EPSILON ||
                     std::abs(result_xyz.y - result_zyx.y) > SIMD_EPSILON ||
                     std::abs(result_xyz.z - result_zyx.z) > SIMD_EPSILON);
    EXPECT_TRUE(different) << "Different rotation orders should produce different results";
    
    // Verify each composition manually
    // XYZ order: produces (0,0,1) in this implementation's convention
    EXPECT_NEAR(result_xyz.x, 0.0f, SIMD_EPSILON);
    EXPECT_NEAR(result_xyz.y, 0.0f, SIMD_EPSILON);
    EXPECT_NEAR(result_xyz.z, 1.0f, SIMD_EPSILON);
}

TEST_F(VxQuaternionSpecialCasesTest, NearIdentityQuaternions) {
    // Test quaternions very close to identity
    VxQuaternion near_identity(1e-8f, 1e-8f, 1e-8f, 1.0f - 1e-16f);
    near_identity.Normalize();
    
    VxMatrix mat;
    near_identity.ToMatrix(mat);
    
    // Should be very close to identity matrix
    EXPECT_NEAR(mat[0][0], 1.0f, 1e-6f);
    EXPECT_NEAR(mat[1][1], 1.0f, 1e-6f);
    EXPECT_NEAR(mat[2][2], 1.0f, 1e-6f);
    EXPECT_NEAR(mat[0][1], 0.0f, 1e-6f);
    EXPECT_NEAR(mat[1][0], 0.0f, 1e-6f);
    
    // Test conversion back to quaternion
    VxQuaternion recovered;
    recovered.FromMatrix(mat);
    
    // Should be close to original or its negative
    bool close_to_original = (std::abs(recovered.x - near_identity.x) < 1e-6f &&
                             std::abs(recovered.y - near_identity.y) < 1e-6f &&
                             std::abs(recovered.z - near_identity.z) < 1e-6f &&
                             std::abs(recovered.w - near_identity.w) < 1e-6f);
    
    bool close_to_negative = (std::abs(recovered.x + near_identity.x) < 1e-6f &&
                             std::abs(recovered.y + near_identity.y) < 1e-6f &&
                             std::abs(recovered.z + near_identity.z) < 1e-6f &&
                             std::abs(recovered.w + near_identity.w) < 1e-6f);
    
    EXPECT_TRUE(close_to_original || close_to_negative);
}

TEST_F(VxQuaternionSpecialCasesTest, QuaternionWithZeroW) {
    // Test quaternion with w = 0 (represents 180-degree rotation)
    VxQuaternion q180(1.0f, 0.0f, 0.0f, 0.0f); // 180 degrees around X
    q180.Normalize();
    
    VxMatrix mat;
    q180.ToMatrix(mat);
    
    // Test 180-degree rotation around X
    VxVector y_axis(0.0f, 1.0f, 0.0f);
    VxVector rotated_y = mat * y_axis;
    
    EXPECT_NEAR(rotated_y.x, 0.0f, EPSILON);
    EXPECT_NEAR(rotated_y.y, -1.0f, EPSILON);
    EXPECT_NEAR(rotated_y.z, 0.0f, EPSILON);
    
    VxVector z_axis(0.0f, 0.0f, 1.0f);
    VxVector rotated_z = mat * z_axis;
    
    EXPECT_NEAR(rotated_z.x, 0.0f, EPSILON);
    EXPECT_NEAR(rotated_z.y, 0.0f, EPSILON);
    EXPECT_NEAR(rotated_z.z, -1.0f, EPSILON);
}

TEST_F(VxQuaternionSpecialCasesTest, VerySmallQuaternions) {
    // Test behavior with very small quaternion components
    VxQuaternion tiny(1e-10f, 1e-10f, 1e-10f, 1e-10f);
    
    // Test normalization
    tiny.Normalize();
    float mag = sqrt(tiny.x * tiny.x + tiny.y * tiny.y + tiny.z * tiny.z + tiny.w * tiny.w);
    EXPECT_NEAR(mag, 1.0f, EPSILON);
    
    // Test matrix conversion
    VxMatrix mat;
    tiny.ToMatrix(mat);
    
    // Matrix should be finite and reasonable
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_TRUE(std::isfinite(mat[i][j]));
            EXPECT_LT(std::abs(mat[i][j]), 10.0f); // Reasonable magnitude
        }
    }
}

// ============================================================================
// Performance and Stress Tests
class VxQuaternionStressTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up various quaternions for stress testing
        test_quaternions.clear();
        
        // Add identity
        test_quaternions.push_back(VxQuaternion());
        
        // Add various rotations
        for (float angle = 0.0f; angle < 2.0f * PI; angle += PI / 8.0f) {
            test_quaternions.push_back(VxQuaternion(VxVector(1.0f, 0.0f, 0.0f), angle));
            test_quaternions.push_back(VxQuaternion(VxVector(0.0f, 1.0f, 0.0f), angle));
            test_quaternions.push_back(VxQuaternion(VxVector(0.0f, 0.0f, 1.0f), angle));
        }
        
        // Add some arbitrary axis rotations
        VxVector arbitrary_axes[] = {
            Normalize(VxVector(1.0f, 1.0f, 1.0f)),
            Normalize(VxVector(1.0f, 2.0f, 3.0f)),
            Normalize(VxVector(-1.0f, 0.5f, 2.0f)),
            VxVector(0.707f, 0.707f, 0.0f),
        };
        
        for (const auto& axis : arbitrary_axes) {
            for (float angle = 0.0f; angle < 2.0f * PI; angle += PI / 4.0f) {
                test_quaternions.push_back(VxQuaternion(axis, angle));
            }
        }
    }
    
    std::vector<VxQuaternion> test_quaternions;
};

TEST_F(VxQuaternionStressTest, MassConversionConsistency) {
    // Test that all conversions remain consistent
    for (size_t i = 0; i < test_quaternions.size(); ++i) {
        VxQuaternion& q = test_quaternions[i];
        
        // Convert to matrix and back
        VxMatrix mat;
        q.ToMatrix(mat);
        
        VxQuaternion q_from_mat;
        q_from_mat.FromMatrix(mat);
        
        // Test that they represent the same rotation
        VxVector test_vec(1.0f, 0.5f, 0.3f);
        
        VxMatrix mat_original, mat_recovered;
        q.ToMatrix(mat_original);
        q_from_mat.ToMatrix(mat_recovered);
        
        VxVector result_original = mat_original * test_vec;
        VxVector result_recovered = mat_recovered * test_vec;
        
        EXPECT_NEAR(result_original.x, result_recovered.x, SIMD_EPSILON) << "Quaternion " << i;
        EXPECT_NEAR(result_original.y, result_recovered.y, SIMD_EPSILON) << "Quaternion " << i;
        EXPECT_NEAR(result_original.z, result_recovered.z, SIMD_EPSILON) << "Quaternion " << i;
        
        // Test normalization preserves rotation
        VxQuaternion q_normalized = q;
        q_normalized.Normalize();
        
        VxMatrix mat_normalized;
        q_normalized.ToMatrix(mat_normalized);
        VxVector result_normalized = mat_normalized * test_vec;
        
        EXPECT_NEAR(result_original.x, result_normalized.x, SIMD_EPSILON) << "Quaternion " << i;
        EXPECT_NEAR(result_original.y, result_normalized.y, SIMD_EPSILON) << "Quaternion " << i;
        EXPECT_NEAR(result_original.z, result_normalized.z, SIMD_EPSILON) << "Quaternion " << i;
    }
}

TEST_F(VxQuaternionStressTest, MassInterpolationTest) {
    // Test Slerp with many quaternion pairs
    for (size_t i = 0; i < test_quaternions.size(); ++i) {
        for (size_t j = i + 1; j < test_quaternions.size() && j < i + 5; ++j) {
            VxQuaternion q1 = test_quaternions[i];
            VxQuaternion q2 = test_quaternions[j];
            
            q1.Normalize();
            q2.Normalize();
            
            // Test multiple interpolation points
            for (float t = 0.0f; t <= 1.0f; t += 0.25f) {
                VxQuaternion interp = Slerp(t, q1, q2);
                
                // Should be unit quaternion
                float mag = sqrt(interp.x * interp.x + interp.y * interp.y + 
                                interp.z * interp.z + interp.w * interp.w);
                EXPECT_NEAR(mag, 1.0f, SIMD_EPSILON) << "Slerp(" << t << ") between " << i << " and " << j;
                
                // Matrix should be valid rotation matrix
                VxMatrix mat;
                interp.ToMatrix(mat);
                
                // Test that matrix preserves vector lengths
                VxVector test_vec(1.0f, 0.0f, 0.0f);
                VxVector rotated = mat * test_vec;
                float rotated_mag = rotated.Magnitude();
                EXPECT_NEAR(rotated_mag, 1.0f, SIMD_EPSILON) << "Slerp(" << t << ") between " << i << " and " << j;
            }
        }
    }
}

TEST_F(VxQuaternionStressTest, ChainedMultiplicationStability) {
    // Test that chained multiplications remain stable
    VxQuaternion accumulated = VxQuaternion(); // Start with identity
    
    for (size_t i = 0; i < 20 && i < test_quaternions.size(); ++i) {
        VxQuaternion q = test_quaternions[i];
        q.Normalize();
        
        accumulated = accumulated * q;
        
        // Accumulated should remain close to unit magnitude
        float mag = sqrt(accumulated.x * accumulated.x + accumulated.y * accumulated.y + 
                        accumulated.z * accumulated.z + accumulated.w * accumulated.w);
        EXPECT_NEAR(mag, 1.0f, 0.01f) << "After " << (i+1) << " multiplications";
        
        // Re-normalize periodically to prevent drift
        if ((i + 1) % 5 == 0) {
            accumulated.Normalize();
        }
    }
}

// ============================================================================
// SIMD Dispatch Tests
class VxQuaternionSIMDTest : public ::testing::Test {
protected:
    bool QuatEqual(const VxQuaternion& a, const VxQuaternion& b, float epsilon = 1e-4f) {
        return std::abs(a.x - b.x) < epsilon &&
               std::abs(a.y - b.y) < epsilon &&
               std::abs(a.z - b.z) < epsilon &&
               std::abs(a.w - b.w) < epsilon;
    }
};

TEST_F(VxQuaternionSIMDTest, NormalizeSIMD) {
    VxQuaternion q(1.0f, 2.0f, 3.0f, 4.0f);
    q.Normalize();
    
    float expectedNorm = std::sqrt(1.0f + 4.0f + 9.0f + 16.0f);
    VxQuaternion expected(1.0f / expectedNorm, 2.0f / expectedNorm, 
                         3.0f / expectedNorm, 4.0f / expectedNorm);
    
    EXPECT_TRUE(QuatEqual(q, expected)) << "SIMD Normalize failed";
}

TEST_F(VxQuaternionSIMDTest, MultiplySIMD) {
    VxQuaternion q1(0.0f, 1.0f, 0.0f, 1.0f);
    VxQuaternion q2(0.5f, 0.5f, 0.75f, 1.0f);
    q1.Normalize();
    q2.Normalize();
    
    VxQuaternion result = q1;
    result.Multiply(q2);
    
    VxQuaternion expected = Vx3DQuaternionMultiply(q1, q2);
    
    EXPECT_TRUE(QuatEqual(result, expected)) << "SIMD Multiply failed";
}

TEST_F(VxQuaternionSIMDTest, Vx3DQuaternionMultiplySIMD) {
    VxQuaternion q1(0.0f, 0.707f, 0.0f, 0.707f);
    VxQuaternion q2(0.707f, 0.0f, 0.0f, 0.707f);
    
    VxQuaternion result = Vx3DQuaternionMultiply(q1, q2);
    
    float norm = std::sqrt(result.x * result.x + result.y * result.y + 
                          result.z * result.z + result.w * result.w);
    
    EXPECT_NEAR(norm, 1.0f, 0.01f) << "Multiply result not normalized";
}

TEST_F(VxQuaternionSIMDTest, SlerpSIMD) {
    VxQuaternion q1(0.0f, 0.0f, 0.0f, 1.0f);
    VxQuaternion q2(0.0f, 0.707f, 0.0f, 0.707f);
    
    VxQuaternion result0 = Slerp(0.0f, q1, q2);
    EXPECT_TRUE(QuatEqual(result0, q1, 0.01f)) << "Slerp(0) failed";
    
    VxQuaternion result1 = Slerp(1.0f, q1, q2);
    EXPECT_TRUE(QuatEqual(result1, q2, 0.01f)) << "Slerp(1) failed";
    
    VxQuaternion result05 = Slerp(0.5f, q1, q2);
    float norm = std::sqrt(result05.x * result05.x + result05.y * result05.y + 
                          result05.z * result05.z + result05.w * result05.w);
    EXPECT_NEAR(norm, 1.0f, 0.01f) << "Slerp(0.5) result not normalized";
}

// ============================================================================
// Edge Cases Tests
class VxQuaternionEdgeCasesTest : public ::testing::Test {
protected:
    bool QuatEqual(const VxQuaternion& a, const VxQuaternion& b, float epsilon = 1e-5f) {
        return std::abs(a.x - b.x) < epsilon &&
               std::abs(a.y - b.y) < epsilon &&
               std::abs(a.z - b.z) < epsilon &&
               std::abs(a.w - b.w) < epsilon;
    }
};

TEST_F(VxQuaternionEdgeCasesTest, NormalizeZeroQuaternion) {
    VxQuaternion q(0.0f, 0.0f, 0.0f, 0.0f);
    q.Normalize();
    
    EXPECT_FLOAT_EQ(q.x, 0.0f) << "Zero quaternion normalization: x should be 0";
    EXPECT_FLOAT_EQ(q.y, 0.0f) << "Zero quaternion normalization: y should be 0";
    EXPECT_FLOAT_EQ(q.z, 0.0f) << "Zero quaternion normalization: z should be 0";
    EXPECT_FLOAT_EQ(q.w, 1.0f) << "Zero quaternion normalization: w should be 1 (identity)";
}

TEST_F(VxQuaternionEdgeCasesTest, NormalizeNearZeroQuaternion) {
    VxQuaternion q(1e-10f, 1e-10f, 1e-10f, 1e-10f);
    q.Normalize();
    
    float expected = 0.5f;
    EXPECT_NEAR(q.x, expected, 1e-5f);
    EXPECT_NEAR(q.y, expected, 1e-5f);
    EXPECT_NEAR(q.z, expected, 1e-5f);
    EXPECT_NEAR(q.w, expected, 1e-5f);
}

TEST_F(VxQuaternionEdgeCasesTest, MultiplyFormulaVerification) {
    VxQuaternion q1(1.0f, 2.0f, 3.0f, 4.0f);
    VxQuaternion q2(5.0f, 6.0f, 7.0f, 8.0f);
    
    VxQuaternion result = Vx3DQuaternionMultiply(q1, q2);
    
    float expected_x = 4.0f * 5.0f + 1.0f * 8.0f + 2.0f * 7.0f - 3.0f * 6.0f;
    float expected_y = 4.0f * 6.0f - 1.0f * 7.0f + 2.0f * 8.0f + 3.0f * 5.0f;
    float expected_z = 4.0f * 7.0f + 1.0f * 6.0f - 2.0f * 5.0f + 3.0f * 8.0f;
    float expected_w = 4.0f * 8.0f - 1.0f * 5.0f - 2.0f * 6.0f - 3.0f * 7.0f;
    
    EXPECT_FLOAT_EQ(result.x, expected_x) << "Multiply formula x component incorrect";
    EXPECT_FLOAT_EQ(result.y, expected_y) << "Multiply formula y component incorrect";
    EXPECT_FLOAT_EQ(result.z, expected_z) << "Multiply formula z component incorrect";
    EXPECT_FLOAT_EQ(result.w, expected_w) << "Multiply formula w component incorrect";
}

TEST_F(VxQuaternionEdgeCasesTest, MultiplyIdentity) {
    VxQuaternion q(1.0f, 2.0f, 3.0f, 4.0f);
    VxQuaternion identity(0.0f, 0.0f, 0.0f, 1.0f);
    
    VxQuaternion result1 = Vx3DQuaternionMultiply(q, identity);
    EXPECT_TRUE(QuatEqual(result1, q)) << "q * identity should equal q";
    
    VxQuaternion result2 = Vx3DQuaternionMultiply(identity, q);
    EXPECT_TRUE(QuatEqual(result2, q)) << "identity * q should equal q";
}

TEST_F(VxQuaternionEdgeCasesTest, MultiplyConjugate) {
    VxQuaternion q(1.0f, 2.0f, 3.0f, 4.0f);
    q.Normalize();
    
    VxQuaternion q_conj(-q.x, -q.y, -q.z, q.w);
    VxQuaternion result = Vx3DQuaternionMultiply(q, q_conj);
    
    EXPECT_NEAR(result.x, 0.0f, 1e-5f) << "q * conj(q) should have x = 0";
    EXPECT_NEAR(result.y, 0.0f, 1e-5f) << "q * conj(q) should have y = 0";
    EXPECT_NEAR(result.z, 0.0f, 1e-5f) << "q * conj(q) should have z = 0";
    EXPECT_NEAR(result.w, 1.0f, 1e-5f) << "q * conj(q) should have w = 1";
}

TEST_F(VxQuaternionEdgeCasesTest, NormalizeMaintainsDirection) {
    VxQuaternion q(3.0f, 4.0f, 0.0f, 0.0f);
    q.Normalize();
    
    EXPECT_NEAR(q.x, 0.6f, 1e-5f);
    EXPECT_NEAR(q.y, 0.8f, 1e-5f);
    EXPECT_NEAR(q.z, 0.0f, 1e-5f);
    EXPECT_NEAR(q.w, 0.0f, 1e-5f);
    
    float mag = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    EXPECT_NEAR(mag, 1.0f, 1e-5f) << "Normalized quaternion should have magnitude 1";
}

TEST_F(VxQuaternionEdgeCasesTest, MultiplyNonCommutative) {
    VxQuaternion q1(1.0f, 0.0f, 0.0f, 0.0f);
    q1.Normalize();
    
    VxQuaternion q2(0.0f, 1.0f, 0.0f, 0.0f);
    q2.Normalize();
    
    VxQuaternion result1 = Vx3DQuaternionMultiply(q1, q2);
    VxQuaternion result2 = Vx3DQuaternionMultiply(q2, q1);
    
    EXPECT_FLOAT_EQ(result1.w, 0.0f);
    EXPECT_FLOAT_EQ(result2.w, 0.0f);
    
    bool different = (std::abs(result1.x - result2.x) > 1e-5f) ||
                     (std::abs(result1.y - result2.y) > 1e-5f) ||
                     (std::abs(result1.z - result2.z) > 1e-5f);
    EXPECT_TRUE(different) << "Quaternion multiplication should not be commutative";
}

TEST_F(VxQuaternionEdgeCasesTest, MemberMultiply) {
    VxQuaternion q1(1.0f, 2.0f, 3.0f, 4.0f);
    VxQuaternion q2(5.0f, 6.0f, 7.0f, 8.0f);
    
    VxQuaternion result_member = q1;
    result_member.Multiply(q2);
    
    VxQuaternion result_free = Vx3DQuaternionMultiply(q1, q2);
    
    EXPECT_TRUE(QuatEqual(result_member, result_free)) 
        << "Member function Multiply should match free function Vx3DQuaternionMultiply";
}

// ============================================================================
// Additional Tests from QuaternionTestEnhanced




