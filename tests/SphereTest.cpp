#include "gtest/gtest.h"
#include "VxMath.h"

// Test fixture for VxSphere tests
class VxSphereTest : public ::testing::Test {
protected:
    // A sphere centered at (10, 20, 30) with a radius of 5
    VxSphere sphere1{ VxVector(10.0f, 20.0f, 30.0f), 5.0f };

    // A default-constructed sphere
    VxSphere defaultSphere;
};

// Test the default constructor
TEST_F(VxSphereTest, DefaultConstructor) {
    // A default-constructed sphere has undefined members, so we just ensure it can be created.
    // The members are public, so we can set them to known values for predictable tests later.
    defaultSphere.Center() = VxVector(0.0f, 0.0f, 0.0f);
    defaultSphere.Radius() = 1.0f;
    EXPECT_EQ(defaultSphere.Center(), VxVector(0.0f, 0.0f, 0.0f));
    EXPECT_EQ(defaultSphere.Radius(), 1.0f);
}

// Test the constructor that takes a center and a radius
TEST_F(VxSphereTest, CenterRadiusConstructor) {
    VxVector center(1.0f, 2.0f, 3.0f);
    float radius = 10.0f;
    VxSphere s(center, radius);

    // Verify that the center and radius are set correctly
    EXPECT_EQ(s.Center(), center);
    EXPECT_EQ(s.Radius(), radius);
}

// Test the accessors for Center()
TEST_F(VxSphereTest, CenterAccessors) {
    // Test const accessor
    const VxVector& centerConst = sphere1.Center();
    EXPECT_EQ(centerConst, VxVector(10.0f, 20.0f, 30.0f));

    // Test non-const accessor for modification
    sphere1.Center() = VxVector(0.0f, 0.0f, 0.0f);
    EXPECT_EQ(sphere1.Center(), VxVector(0.0f, 0.0f, 0.0f));
}

// Test the accessors for Radius()
TEST_F(VxSphereTest, RadiusAccessors) {
    // Test const accessor
    const float& radiusConst = sphere1.Radius();
    EXPECT_EQ(radiusConst, 5.0f);

    // Test non-const accessor for modification
    sphere1.Radius() = 10.0f;
    EXPECT_EQ(sphere1.Radius(), 10.0f);
}

// Test IsPointInside method
TEST_F(VxSphereTest, IsPointInside) {
    // Point clearly inside the sphere
    VxVector pointInside(11.0f, 21.0f, 31.0f);
    EXPECT_TRUE(sphere1.IsPointInside(pointInside));

    // Point clearly outside the sphere
    VxVector pointOutside(20.0f, 20.0f, 30.0f);
    EXPECT_FALSE(sphere1.IsPointInside(pointOutside));

    // Point exactly on the surface of the sphere
    // The distance from (10,20,30) to (15,20,30) is exactly 5 (the radius)
    VxVector pointOnSurface(15.0f, 20.0f, 30.0f);
    EXPECT_TRUE(sphere1.IsPointInside(pointOnSurface));

    // Point at the center of the sphere
    VxVector pointAtCenter = sphere1.Center();
    EXPECT_TRUE(sphere1.IsPointInside(pointAtCenter));
}

// Test IsBoxTotallyInside method
TEST_F(VxSphereTest, IsBoxTotallyInside) {
    // A small box centered within the sphere, so it's fully contained
    VxBbox boxTotallyInside(
        VxVector(9.0f, 19.0f, 29.0f),
        VxVector(11.0f, 21.0f, 31.0f)
    );
    EXPECT_TRUE(sphere1.IsBoxTotallyInside(boxTotallyInside));

    // A box that is partially inside and partially outside the sphere
    VxBbox boxPartiallyInside(
        VxVector(14.0f, 20.0f, 30.0f), // One corner is inside
        VxVector(16.0f, 22.0f, 32.0f)  // Other corner is outside
    );
    EXPECT_FALSE(sphere1.IsBoxTotallyInside(boxPartiallyInside));

    // A box that contains the sphere, so it's not totally inside
    VxBbox boxContainingSphere(
        VxVector(0.0f, 10.0f, 20.0f),
        VxVector(20.0f, 30.0f, 40.0f)
    );
    EXPECT_FALSE(sphere1.IsBoxTotallyInside(boxContainingSphere));
    
    // A box that is completely outside the sphere
    VxBbox boxTotallyOutside(
        VxVector(100.0f, 100.0f, 100.0f),
        VxVector(101.0f, 101.0f, 101.0f)
    );
    EXPECT_FALSE(sphere1.IsBoxTotallyInside(boxTotallyOutside));

    // A box that is tangent to the sphere's surface from the inside
    VxBbox boxTangentInside(
        sphere1.Center() - VxVector(1.0f, 1.0f, 1.0f),
        sphere1.Center() + VxVector(1.0f, 1.0f, 1.0f)
    );
    // This box's farthest corner has a squared distance of 1*1+1*1+1*1=3 from the center, which is < 25 (radius^2)
    EXPECT_TRUE(sphere1.IsBoxTotallyInside(boxTangentInside));
}


// Test IsPointOnSurface method
TEST_F(VxSphereTest, IsPointOnSurface) {
    // Point clearly inside the sphere, not on surface
    VxVector pointInside(11.0f, 21.0f, 31.0f);
    EXPECT_FALSE(sphere1.IsPointOnSurface(pointInside));

    // Point clearly outside the sphere, not on surface
    VxVector pointOutside(20.0f, 20.0f, 30.0f);
    EXPECT_FALSE(sphere1.IsPointOnSurface(pointOutside));

    // Point exactly on the surface. Uses a small epsilon for float comparison.
    VxVector pointOnSurface(15.0f, 20.0f, 30.0f);
    EXPECT_TRUE(sphere1.IsPointOnSurface(pointOnSurface));
}

// Test the equality operator
TEST_F(VxSphereTest, EqualityOperator) {
    VxSphere sphereA(VxVector(1.0f, 2.0f, 3.0f), 4.0f);
    VxSphere sphereB(VxVector(1.0f, 2.0f, 3.0f), 4.0f); // Identical to A
    VxSphere sphereC(VxVector(1.0f, 2.0f, 3.0f), 5.0f); // Different radius
    VxSphere sphereD(VxVector(9.0f, 2.0f, 3.0f), 4.0f); // Different center

    // Test for equality
    EXPECT_TRUE(sphereA == sphereB);

    // Test for inequality due to radius
    EXPECT_FALSE(sphereA == sphereC);

    // Test for inequality due to center
    EXPECT_FALSE(sphereA == sphereD);

    // Self-equality
    EXPECT_TRUE(sphereA == sphereA);
}