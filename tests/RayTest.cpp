#include "gtest/gtest.h"
#include "VxMath.h"

// Test fixture for VxRay tests
class VxRayTest : public ::testing::Test {
protected:
    // Pre-defined vectors for consistent testing
    const VxVector zero_origin{0.0f, 0.0f, 0.0f};
    const VxVector p1{1.0f, 2.0f, 3.0f};
    const VxVector p2{5.0f, 6.0f, 7.0f};
    const VxVector dir_x{1.0f, 0.0f, 0.0f};
    const VxVector dir_z{0.0f, 0.0f, 1.0f};
};

// Test the default constructor.
// While it does nothing, this ensures it can be instantiated.
TEST_F(VxRayTest, DefaultConstructor) {
    VxRay ray;
    // The members are uninitialized, so we just check for successful creation.
    SUCCEED();
}

// Test the constructor that takes a start and end point.
TEST_F(VxRayTest, ConstructorFromStartEnd) {
    VxRay ray(p1, p2);
    
    // The origin should be the start point.
    EXPECT_EQ(ray.GetOrigin(), p1);
    
    // The direction should be the vector from start to end.
    VxVector expected_direction = p2 - p1; // (4, 4, 4)
    EXPECT_EQ(ray.GetDirection(), expected_direction);
}

// Test the constructor that takes a start point and a direction vector.
TEST_F(VxRayTest, ConstructorFromStartDirection) {
    VxRay ray(p1, dir_x, nullptr); // The dummy int* selects this overload
    
    // The origin should be the start point.
    EXPECT_EQ(ray.GetOrigin(), p1);
    
    // The direction should be the provided direction vector.
    EXPECT_EQ(ray.GetDirection(), dir_x);
}

// Test the copy constructor (implicitly defined by the compiler).
TEST_F(VxRayTest, CopyConstructor) {
    VxRay original_ray(p1, dir_x, nullptr);
    VxRay copied_ray(original_ray);
    
    // The copied ray should be identical to the original.
    EXPECT_EQ(copied_ray.GetOrigin(), original_ray.GetOrigin());
    EXPECT_EQ(copied_ray.GetDirection(), original_ray.GetDirection());
    EXPECT_EQ(copied_ray, original_ray);
}

// Test the assignment operator (implicitly defined by the compiler).
TEST_F(VxRayTest, AssignmentOperator) {
    VxRay original_ray(p1, dir_x, nullptr);
    VxRay assigned_ray;
    assigned_ray = original_ray;
    
    // The assigned ray should be identical to the original.
    EXPECT_EQ(assigned_ray.GetOrigin(), original_ray.GetOrigin());
    EXPECT_EQ(assigned_ray.GetDirection(), original_ray.GetDirection());
    
    // Test self-assignment
    assigned_ray = assigned_ray;
    EXPECT_EQ(assigned_ray, original_ray);
}

// Test the equality operator.
TEST_F(VxRayTest, EqualityOperator) {
    VxRay ray1(p1, dir_x, nullptr);
    VxRay ray2(p1, dir_x, nullptr);
    VxRay ray3(p2, dir_x, nullptr); // Different origin
    VxRay ray4(p1, dir_z, nullptr); // Different direction
    
    EXPECT_TRUE(ray1 == ray2);
    EXPECT_FALSE(ray1 == ray3);
    EXPECT_FALSE(ray1 == ray4);
}

// Test the GetOrigin and GetDirection accessors, both const and non-const.
TEST_F(VxRayTest, Accessors) {
    VxRay ray(p1, p2);
    
    // Test non-const accessors
    ray.GetOrigin().x = 99.0f;
    ray.GetDirection().y = -99.0f;
    EXPECT_EQ(ray.m_Origin.x, 99.0f);
    EXPECT_EQ(ray.m_Direction.y, -99.0f);

    // Test const accessors
    const VxRay& const_ray = ray;
    EXPECT_EQ(const_ray.GetOrigin().x, 99.0f);
    EXPECT_EQ(const_ray.GetDirection().y, -99.0f);
}

// Test the Interpolate method to find a point along the ray.
TEST_F(VxRayTest, Interpolate) {
    VxRay ray(p1, p2); // Direction is (4, 4, 4)
    VxVector point_on_ray;

    // Interpolate at the start point (t=0)
    ray.Interpolate(point_on_ray, 0.0f);
    EXPECT_EQ(point_on_ray, p1);

    // Interpolate at the end point (t=1)
    ray.Interpolate(point_on_ray, 1.0f);
    EXPECT_EQ(point_on_ray, p2);

    // Interpolate at the midpoint (t=0.5)
    ray.Interpolate(point_on_ray, 0.5f);
    VxVector midpoint = p1 + (p2 - p1) * 0.5f; // (1,2,3) + (2,2,2) = (3,4,5)
    EXPECT_EQ(point_on_ray, midpoint);
    
    // Interpolate beyond the end point (t=2.0)
    ray.Interpolate(point_on_ray, 2.0f);
    VxVector beyond_point = p1 + (p2 - p1) * 2.0f;
    EXPECT_EQ(point_on_ray, beyond_point);
}

// Test the calculation of the squared distance from a point to the ray's line.
TEST_F(VxRayTest, SquareDistance) {
    // A ray along the X-axis starting at the origin
    VxRay ray(zero_origin, dir_x, nullptr);
    
    // A point directly on the line
    VxVector point_on_line(10.0f, 0.0f, 0.0f);
    EXPECT_NEAR(ray.SquareDistance(point_on_line), 0.0f, 1e-6);
    
    // A point off the line
    VxVector point_off_line(5.0f, 3.0f, 4.0f); // Perpendicular distance is sqrt(3^2 + 4^2) = 5
    // Squared distance should be 25
    EXPECT_NEAR(ray.SquareDistance(point_off_line), 25.0f, 1e-6);

    // A point at the origin of the ray
    EXPECT_NEAR(ray.SquareDistance(zero_origin), 0.0f, 1e-6);
}

// Test the calculation of the distance from a point to the ray's line.
TEST_F(VxRayTest, Distance) {
    // A ray along the X-axis starting at the origin
    VxRay ray(zero_origin, dir_x, nullptr);

    // A point off the line
    VxVector point_off_line(5.0f, 3.0f, 4.0f); // Perpendicular distance is sqrt(3^2 + 4^2) = 5
    EXPECT_NEAR(ray.Distance(point_off_line), 5.0f, 1e-6);

    // A point directly on the line
    VxVector point_on_line(10.0f, 0.0f, 0.0f);
    EXPECT_NEAR(ray.Distance(point_on_line), 0.0f, 1e-6);
}

// Test the transformation of a ray by a matrix.
TEST_F(VxRayTest, Transform) {
    // A simple ray along the X axis starting at p1
    VxRay source_ray(p1, dir_x, nullptr);
    VxRay dest_ray;

    // Case 1: Identity matrix (no change)
    VxMatrix identity_matrix = VxMatrix::Identity();
    source_ray.Transform(dest_ray, identity_matrix);
    EXPECT_EQ(dest_ray, source_ray);

    // Case 2: Translation matrix
    VxMatrix translation_matrix = VxMatrix::Identity();
    translation_matrix[3].Set(10.0f, 20.0f, 30.0f);
    source_ray.Transform(dest_ray, translation_matrix);

    // The origin should be translated.
    VxVector expected_translated_origin = p1 + VxVector(10.0f, 20.0f, 30.0f);
    EXPECT_EQ(dest_ray.GetOrigin(), expected_translated_origin);
    // The direction vector should NOT be affected by a pure translation.
    EXPECT_EQ(dest_ray.GetDirection(), dir_x);

    // Case 3: Rotation matrix (90 degrees around Z-axis)
    VxMatrix rotation_matrix;
    Vx3DMatrixFromRotation(rotation_matrix, dir_z, PI / 2.0f);
    source_ray.Transform(dest_ray, rotation_matrix);

    // The origin p1(1,2,3) should be rotated.
    VxVector expected_rotated_origin;
    Vx3DMultiplyMatrixVector(&expected_rotated_origin, rotation_matrix, &p1);

    // The direction (1,0,0) should be rotated to (0,1,0).
    VxVector expected_rotated_dir;
    Vx3DRotateVector(&expected_rotated_dir, rotation_matrix, &dir_x);

    EXPECT_NEAR(dest_ray.GetOrigin().x, expected_rotated_origin.x, 1e-6);
    EXPECT_NEAR(dest_ray.GetOrigin().y, expected_rotated_origin.y, 1e-6);
    EXPECT_NEAR(dest_ray.GetOrigin().z, expected_rotated_origin.z, 1e-6);

    EXPECT_NEAR(dest_ray.GetDirection().x, expected_rotated_dir.x, 1e-6);
    EXPECT_NEAR(dest_ray.GetDirection().y, expected_rotated_dir.y, 1e-6);
    EXPECT_NEAR(dest_ray.GetDirection().z, expected_rotated_dir.z, 1e-6);

    // Case 4: Combined transformation (Scale, then Rotate, then Translate)
    VxMatrix scale_matrix = VxMatrix::Identity();
    scale_matrix[0][0] = 2.0f; // Scale X by 2

    VxMatrix combo_matrix;
    // Build transform: M = T * R * S
    Vx3DMultiplyMatrix(combo_matrix, rotation_matrix, scale_matrix);
    Vx3DMultiplyMatrix(combo_matrix, translation_matrix, combo_matrix);

    source_ray.Transform(dest_ray, combo_matrix);

    // The origin is fully transformed by the combined matrix.
    VxVector expected_new_origin;
    Vx3DMultiplyMatrixVector(&expected_new_origin, combo_matrix, &p1);

    // The direction is transformed by the rotation and scale parts only (no translation).
    VxVector expected_new_dir;
    Vx3DRotateVector(&expected_new_dir, combo_matrix, &dir_x);
    
    EXPECT_NEAR(dest_ray.GetOrigin().x, expected_new_origin.x, 1e-5);
    EXPECT_NEAR(dest_ray.GetOrigin().y, expected_new_origin.y, 1e-5);
    EXPECT_NEAR(dest_ray.GetOrigin().z, expected_new_origin.z, 1e-5);

    EXPECT_NEAR(dest_ray.GetDirection().x, expected_new_dir.x, 1e-5);
    EXPECT_NEAR(dest_ray.GetDirection().y, expected_new_dir.y, 1e-5);
    EXPECT_NEAR(dest_ray.GetDirection().z, expected_new_dir.z, 1e-5);
}