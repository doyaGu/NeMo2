#include "gtest/gtest.h"
#include "VxMath.h"

// Test fixture for VxPlane tests
class VxPlaneTest : public ::testing::Test {
protected:
    // A plane facing positive Y, passing through (0, 1, 0)
    // Equation: 0*x + 1*y + 0*z - 1 = 0  => y - 1 = 0
    VxPlane plane_y_up{VxVector(0, 1, 0), -1.0f};

    // A more general plane passing through (1,1,1) with normal (1,1,1)
    // Equation: x + y + z - 3 = 0
    VxPlane general_plane{VxVector(1, 1, 1), -3.0f};
};

// Test all constructors of VxPlane
TEST_F(VxPlaneTest, Construction) {
    // 1. Default constructor
    VxPlane default_plane;
    EXPECT_EQ(default_plane.m_Normal, VxVector(0, 0, 0));
    EXPECT_FLOAT_EQ(default_plane.m_D, 0.0f);

    // 2. Normal and distance constructor
    VxPlane plane_from_nd(VxVector(0, 0, 1), -5.0f);
    EXPECT_EQ(plane_from_nd.m_Normal, VxVector(0, 0, 1));
    EXPECT_FLOAT_EQ(plane_from_nd.m_D, -5.0f);

    // 3. Four floats constructor (a, b, c, d)
    VxPlane plane_from_floats(0, 0, 1, -5.0f);
    EXPECT_EQ(plane_from_floats, plane_from_nd);

    // 4. Normal and point constructor
    VxVector point_on_plane(10, 20, 5); // A point on the z=5 plane
    VxPlane plane_from_np(VxVector(0, 0, 1), point_on_plane);
    // Check if the point itself lies on the generated plane
    EXPECT_NEAR(plane_from_np.Classify(point_on_plane), 0.0f, EPSILON);
    // The normal should be normalized by Create()
    EXPECT_NEAR(plane_from_np.m_Normal.Magnitude(), 1.0f, EPSILON);

    // 5. Three points constructor
    VxVector p1(1, 0, 0), p2(0, 1, 0), p3(0, 0, 1);
    VxPlane plane_from_3p(p1, p2, p3);
    // All three points should lie on the generated plane
    EXPECT_NEAR(plane_from_3p.Classify(p1), 0.0f, EPSILON);
    EXPECT_NEAR(plane_from_3p.Classify(p2), 0.0f, EPSILON);
    EXPECT_NEAR(plane_from_3p.Classify(p3), 0.0f, EPSILON);
    // The normal should be normalized
    EXPECT_NEAR(plane_from_3p.m_Normal.Magnitude(), 1.0f, EPSILON);
}

// Test point classification relative to the plane
TEST_F(VxPlaneTest, ClassifyPoint) {
    // Test with the plane y = 1
    // A point in front of the plane (in direction of normal)
    EXPECT_GT(plane_y_up.Classify(VxVector(5, 2, 5)), 0.0f);

    // A point behind the plane
    EXPECT_LT(plane_y_up.Classify(VxVector(5, 0, 5)), 0.0f);

    // A point exactly on the plane
    EXPECT_NEAR(plane_y_up.Classify(VxVector(5, 1, 5)), 0.0f, EPSILON);
}

// Test AABB (Axis-Aligned Bounding Box) classification
TEST_F(VxPlaneTest, ClassifyBox) {
    // An AABB entirely in front of the plane y = 1
    VxBbox box_in_front(VxVector(0, 2, 0), VxVector(1, 3, 1));
    EXPECT_GT(plane_y_up.Classify(box_in_front), 0.0f);

    // An AABB entirely behind the plane y = 1
    VxBbox box_behind(VxVector(0, -1, 0), VxVector(1, 0, 1));
    EXPECT_LT(plane_y_up.Classify(box_behind), 0.0f);

    // An AABB that straddles the plane y = 1
    VxBbox box_straddling(VxVector(0, 0, 0), VxVector(1, 2, 1));
    // The result for intersecting should be exactly 0.0f
    EXPECT_FLOAT_EQ(plane_y_up.Classify(box_straddling), 0.0f);

    // Test a case where the normal has negative components
    VxPlane plane_neg_normal(VxVector(0, -1, 0), 1.0f); // Plane -y + 1 = 0 => y = 1
    // With a negative normal, points with y > 1 are behind the plane (negative distance).
    EXPECT_LT(plane_neg_normal.Classify(box_in_front), 0.0f);
}

// Test OBB (Oriented Bounding Box) classification
TEST_F(VxPlaneTest, ClassifyBoxWithMatrix) {
    // A simple unit box at the origin
    VxBbox unit_box(VxVector(-0.5f, -0.5f, -0.5f), VxVector(0.5f, 0.5f, 0.5f));
    
    // Case 1: OBB is fully in front
    VxMatrix mat_in_front;
    mat_in_front.SetIdentity();
    mat_in_front[3] = VxVector(0, 3, 0); // Translate box center to (0,3,0)
    EXPECT_GT(plane_y_up.Classify(unit_box, mat_in_front), 0.0f);

    // Case 2: OBB is fully behind
    VxMatrix mat_behind;
    mat_behind.SetIdentity();
    mat_behind[3] = VxVector(0, -3, 0); // Translate box center to (0,-3,0)
    EXPECT_LT(plane_y_up.Classify(unit_box, mat_behind), 0.0f);
    
    // Case 3: OBB is intersecting the plane
    VxMatrix mat_intersecting;
    mat_intersecting.SetIdentity();
    mat_intersecting[3] = VxVector(0, 1, 0); // Center is on the plane
    EXPECT_FLOAT_EQ(plane_y_up.Classify(unit_box, mat_intersecting), 0.0f);

    // Case 4: A rotated OBB that is intersecting
    VxMatrix mat_rotated_intersect;
    Vx3DMatrixFromRotation(mat_rotated_intersect, VxVector::axisZ(), PI / 4.0f); // 45-degree rotation
    mat_rotated_intersect[3] = VxVector(0, 1, 0); // Center is on the plane
    EXPECT_FLOAT_EQ(plane_y_up.Classify(unit_box, mat_rotated_intersect), 0.0f);
}

// Test face classification (all 3 vertices of a triangle)
TEST_F(VxPlaneTest, ClassifyFace) {
    // A face entirely in front of the plane y = 1
    VxVector f1(0, 2, 0), f2(1, 3, 0), f3(0, 2, 1);
    EXPECT_GT(plane_y_up.ClassifyFace(f1, f2, f3), 0.0f);
    
    // A face entirely behind the plane y = 1
    VxVector b1(0, 0, 0), b2(1, -1, 0), b3(0, 0, 1);
    EXPECT_LT(plane_y_up.ClassifyFace(b1, b2, b3), 0.0f);
    
    // A face that straddles the plane y = 1
    VxVector s1(0, 0, 0), s2(1, 2, 0), s3(0, 0, 1);
    EXPECT_FLOAT_EQ(plane_y_up.ClassifyFace(s1, s2, s3), 0.0f);
    
    // Test combinations to cover all branches in the function logic
    // min > 0, d < 0
    EXPECT_FLOAT_EQ(plane_y_up.ClassifyFace(VxVector(0,2,0), VxVector(0,0,0), VxVector(0,3,0)), 0.0f);
    // min < 0, d > 0
    EXPECT_FLOAT_EQ(plane_y_up.ClassifyFace(VxVector(0,0,0), VxVector(0,2,0), VxVector(0,-1,0)), 0.0f);
}

// Test geometric operations on the plane
TEST_F(VxPlaneTest, Operations) {
    // 1. Distance from a point to the plane
    // For y=1 plane, distance to (10, 5, 10) should be |5 - 1| = 4
    EXPECT_FLOAT_EQ(plane_y_up.Distance(VxVector(10, 5, 10)), 4.0f);
    EXPECT_FLOAT_EQ(plane_y_up.Distance(VxVector(10, -3, 10)), 4.0f);

    // 2. Nearest point on the plane
    VxVector p(10, 5, 20);
    VxVector nearest = plane_y_up.NearestPoint(p);
    // For y=1 plane, nearest point to (10,5,20) is (10,1,20)
    EXPECT_EQ(nearest, VxVector(10, 1, 20));
    // The nearest point must lie on the plane
    EXPECT_NEAR(plane_y_up.Classify(nearest), 0.0f, EPSILON);

    // 3. Unary minus operator (plane negation)
    VxPlane negated_plane = -plane_y_up;
    EXPECT_EQ(negated_plane.m_Normal, -plane_y_up.m_Normal);
    EXPECT_FLOAT_EQ(negated_plane.m_D, -plane_y_up.m_D);
    // A point in front of original plane should be behind the new one
    EXPECT_LT(negated_plane.Classify(VxVector(0, 2, 0)), 0.0f);
}

// Test comparison operator
TEST_F(VxPlaneTest, Equality) {
    VxPlane plane1(VxVector(0, 1, 0), -1.0f);
    VxPlane plane2(VxVector(0, 1, 0), -1.0f);
    VxPlane plane3(VxVector(1, 0, 0), -1.0f); // Different normal
    VxPlane plane4(VxVector(0, 1, 0), -2.0f); // Different distance

    EXPECT_TRUE(plane1 == plane2);
    EXPECT_FALSE(plane1 == plane3);
    EXPECT_FALSE(plane1 == plane4);
}