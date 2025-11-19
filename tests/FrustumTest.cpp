#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "VxFrustum.h"
#include "VxIntersect.h"

// Tolerance for SIMD operations - SIMD can have slightly different rounding than scalar operations
constexpr float SIMD_EPSILON = 5e-07f;

// Use a tolerance for floating-point comparisons
const float VEC_TOLERANCE = SIMD_EPSILON;
const float FLOAT_TOLERANCE = SIMD_EPSILON;

// Helper to compare two VxVector objects with a tolerance.
void EXPECT_VECTORS_NEAR(const VxVector& v1, const VxVector& v2, float tolerance = VEC_TOLERANCE) {
    EXPECT_NEAR(v1.x, v2.x, tolerance);
    EXPECT_NEAR(v1.y, v2.y, tolerance);
    EXPECT_NEAR(v1.z, v2.z, tolerance);
}

// Helper to compare two VxPlane objects with a tolerance.
void EXPECT_PLANES_NEAR(const VxPlane& p1, const VxPlane& p2, float tolerance = VEC_TOLERANCE) {
    EXPECT_VECTORS_NEAR(p1.GetNormal(), p2.GetNormal(), tolerance);
    EXPECT_NEAR(p1.m_D, p2.m_D, tolerance);
}

// Test fixture for VxFrustum
class VxFrustumTest : public ::testing::Test {
protected:
    // Standard perspective frustum for testing
    VxFrustum perspectiveFrustum;
    // Orthographic frustum for testing
    VxFrustum orthoFrustum;

    // Parameters for the perspective frustum
    VxVector perspectiveOrigin = VxVector(0, 0, 0);
    VxVector perspectiveDir = VxVector(0, 0, 1);
    VxVector perspectiveUp = VxVector(0, 1, 0);
    VxVector perspectiveRight = VxVector(1, 0, 0);
    float fov = PI / 2.0f; // 90 degrees FOV
    float aspect = 1.0f;
    float nearPlaneDist = 1.0f;
    float farPlaneDist = 100.0f;

    // Parameters for the orthographic frustum
    VxVector orthoOrigin = VxVector(0, 0, -50); // Moved back to see the origin
    VxVector orthoDir = VxVector(0, 0, 1);
    VxVector orthoUp = VxVector(0, 1, 0);
    VxVector orthoRight = VxVector(1, 0, 0);
    // Orthographic uses bounds instead of FOV, which we can simulate via the constructor
    // by providing pre-calculated bounds. Let's make it 10 units wide and high.
    float orthoRBound = 5.0f;
    float orthoUBound = 5.0f;
    float orthoNear = 1.0f;
    float orthoFar = 100.0f;


    void SetUp() override {
        // Initialize the perspective frustum before each test
        perspectiveFrustum = VxFrustum(
            perspectiveOrigin,
            perspectiveRight,
            perspectiveUp,
            perspectiveDir,
            nearPlaneDist,
            farPlaneDist,
            fov,
            aspect);

        // For the ortho frustum, we create it manually as the constructor is perspective-focused.
        // We will set its properties directly.
        orthoFrustum.GetOrigin() = orthoOrigin;
        orthoFrustum.GetRight() = orthoRight;
        orthoFrustum.GetUp() = orthoUp;
        orthoFrustum.GetDir() = orthoDir;
        orthoFrustum.GetRBound() = orthoRBound;
        orthoFrustum.GetUBound() = orthoUBound;
        orthoFrustum.GetDMin() = orthoNear;
        orthoFrustum.GetDMax() = orthoFar;
        orthoFrustum.Update(); // This calculates the planes
    }
};

// Test the constructor and initial state of the frustum.
TEST_F(VxFrustumTest, ConstructorAndInitialization) {
    // Check that the initial properties match the input parameters.
    EXPECT_VECTORS_NEAR(perspectiveFrustum.GetOrigin(), perspectiveOrigin);
    EXPECT_VECTORS_NEAR(perspectiveFrustum.GetDir(), perspectiveDir);
    EXPECT_VECTORS_NEAR(perspectiveFrustum.GetUp(), perspectiveUp);
    EXPECT_VECTORS_NEAR(perspectiveFrustum.GetRight(), perspectiveRight);

    EXPECT_NEAR(perspectiveFrustum.GetDMin(), nearPlaneDist, FLOAT_TOLERANCE);
    EXPECT_NEAR(perspectiveFrustum.GetDMax(), farPlaneDist, FLOAT_TOLERANCE);

    // With FOV = 90 degrees and aspect = 1, the half-width at the near plane should be equal to the near plane distance.
    EXPECT_NEAR(perspectiveFrustum.GetRBound(), nearPlaneDist, FLOAT_TOLERANCE);
    EXPECT_NEAR(perspectiveFrustum.GetUBound(), nearPlaneDist, FLOAT_TOLERANCE);

    // Verify the planes are calculated correctly.
    // Near plane: normal is the frustum direction, point is origin + dir * near_dist
    VxPlane expectedNearPlane(perspectiveDir, perspectiveOrigin + perspectiveDir * nearPlaneDist);
    EXPECT_PLANES_NEAR(perspectiveFrustum.GetNearPlane(), expectedNearPlane);

    // Far plane: normal is -direction, point is origin + dir * far_dist
    VxPlane expectedFarPlane(-perspectiveDir, perspectiveOrigin + perspectiveDir * farPlaneDist);
    EXPECT_PLANES_NEAR(perspectiveFrustum.GetFarPlane(), expectedFarPlane);

    // The other planes (left, right, top, bottom) are angled for a perspective frustum.
    // For a 90-degree FOV, the angle between the direction and the side plane normal's projection is 45 degrees.
    // The normal to the left plane should be a rotation of the 'right' vector around the 'up' axis.
    VxVector leftPlaneNormal = Rotate(perspectiveUp, -perspectiveRight, PI / 4.0f);
    VxPlane expectedLeftPlane(leftPlaneNormal, perspectiveOrigin);
    EXPECT_PLANES_NEAR(perspectiveFrustum.GetLeftPlane(), expectedLeftPlane);

    VxVector rightPlaneNormal = Rotate(perspectiveUp, perspectiveRight, -PI / 4.0f);
    VxPlane expectedRightPlane(rightPlaneNormal, perspectiveOrigin);
    EXPECT_PLANES_NEAR(perspectiveFrustum.GetRightPlane(), expectedRightPlane);
}

// Test the Update method to ensure it recalculates planes correctly.
TEST_F(VxFrustumTest, Update) {
    VxFrustum frustum = perspectiveFrustum;

    // Change a parameter, e.g., the direction
    frustum.GetDir() = VxVector(0, 0, -1);
    frustum.GetRight() = VxVector(-1, 0, 0); // Must update right vector to stay orthogonal

    // The planes should still be the old ones until we call Update()
    VxPlane oldNearPlane = frustum.GetNearPlane();

    // Now update
    frustum.Update();
    VxPlane newNearPlane = frustum.GetNearPlane();

    // The new near plane should be different from the old one
    EXPECT_FALSE(oldNearPlane == newNearPlane);

    // And it should match the new direction
    VxPlane expectedNewNearPlane(frustum.GetDir(), frustum.GetOrigin() + frustum.GetDir() * frustum.GetDMin());
    EXPECT_PLANES_NEAR(newNearPlane, expectedNewNearPlane);
}


// Test point classification against the perspective frustum.
TEST_F(VxFrustumTest, ClassifyPoint) {
    // A point deep inside the frustum
    VxVector pointInside(0, 0, 50.0f);
    EXPECT_EQ(perspectiveFrustum.Classify(pointInside), 0) << "Point deep inside should have no clip flags.";
    EXPECT_TRUE(perspectiveFrustum.IsInside(pointInside));

    // A point just inside the near plane
    VxVector pointJustInside(0, 0, nearPlaneDist + 0.1f);
    EXPECT_EQ(perspectiveFrustum.Classify(pointJustInside), 0) << "Point just inside near plane should have no clip flags.";
    EXPECT_TRUE(perspectiveFrustum.IsInside(pointJustInside));

    // A point just in front of the near plane
    VxVector pointInFront(0, 0, nearPlaneDist - 0.1f);
    EXPECT_EQ(perspectiveFrustum.Classify(pointInFront), VXCLIP_FRONT) << "Point in front of near plane should be clipped.";
    EXPECT_FALSE(perspectiveFrustum.IsInside(pointInFront));

    // A point just behind the far plane
    VxVector pointBehind(0, 0, farPlaneDist + 0.1f);
    EXPECT_EQ(perspectiveFrustum.Classify(pointBehind), VXCLIP_BACK) << "Point behind far plane should be clipped.";
    EXPECT_FALSE(perspectiveFrustum.IsInside(pointBehind));

    // A point to the left of the frustum
    VxVector pointLeft(-20, 0, 50.0f);
    EXPECT_EQ(perspectiveFrustum.Classify(pointLeft), VXCLIP_LEFT) << "Point left of frustum should be clipped.";
    EXPECT_FALSE(perspectiveFrustum.IsInside(pointLeft));

    // A point to the right of the frustum
    VxVector pointRight(20, 0, 50.0f);
    EXPECT_EQ(perspectiveFrustum.Classify(pointRight), VXCLIP_RIGHT) << "Point right of frustum should be clipped.";
    EXPECT_FALSE(perspectiveFrustum.IsInside(pointRight));

    // A point above the frustum
    VxVector pointTop(0, 20, 50.0f);
    EXPECT_EQ(perspectiveFrustum.Classify(pointTop), VXCLIP_TOP) << "Point above frustum should be clipped.";
    EXPECT_FALSE(perspectiveFrustum.IsInside(pointTop));

    // A point below the frustum
    VxVector pointBottom(0, -20, 50.0f);
    EXPECT_EQ(perspectiveFrustum.Classify(pointBottom), VXCLIP_BOTTOM) << "Point below frustum should be clipped.";
    EXPECT_FALSE(perspectiveFrustum.IsInside(pointBottom));

    // A point outside multiple planes (front and left)
    VxVector pointFrontLeft(-20, 0, nearPlaneDist - 0.1f);
    EXPECT_EQ(perspectiveFrustum.Classify(pointFrontLeft), VXCLIP_FRONT | VXCLIP_LEFT) << "Point front-left should have two clip flags.";
}

// Test AABB (Axis-Aligned Bounding Box) classification.
TEST_F(VxFrustumTest, ClassifyAABB) {
    // Box completely inside the frustum
    VxBbox boxInside(VxVector(-1, -1, 10), VxVector(1, 1, 20));
    EXPECT_LT(perspectiveFrustum.Classify(boxInside), 0.0f) << "Box fully inside should return a negative value (no single plane separates).";

    // Box intersecting the near plane
    VxBbox boxIntersectNear(VxVector(-1, -1, 0.5), VxVector(1, 1, 1.5));
    EXPECT_FLOAT_EQ(perspectiveFrustum.Classify(boxIntersectNear), 0.0f) << "Box intersecting should return 0.";

    // Box completely in front of the frustum
    VxBbox boxInFront(VxVector(-1, -1, 0.1), VxVector(1, 1, 0.5));
    EXPECT_GT(perspectiveFrustum.Classify(boxInFront), 0.0f) << "Box fully in front should return positive value.";

    // Box completely behind the frustum
    VxBbox boxBehind(VxVector(-1, -1, 101), VxVector(1, 1, 102));
    EXPECT_GT(perspectiveFrustum.Classify(boxBehind), 0.0f) << "Box fully behind should return positive value.";

    // Box completely to the left of the frustum
    VxBbox boxLeft(VxVector(-20, -1, 10), VxVector(-10, 1, 20));
    EXPECT_GT(perspectiveFrustum.Classify(boxLeft), 0.0f) << "Box fully to the left should return positive value.";

    // Huge box that engulfs the entire frustum
    VxBbox boxEngulfing(VxVector(-200, -200, -10), VxVector(200, 200, 200));
    EXPECT_LT(perspectiveFrustum.Classify(boxEngulfing), 0.0f) << "Engulfing box should be classified as inside.";
}


// Test OBB (Oriented Bounding Box) classification.
TEST_F(VxFrustumTest, ClassifyOBB) {
    // An OBB is an AABB + a transformation matrix.
    // Start with a box that's inside when axis-aligned.
    VxBbox baseBox(VxVector(-1, -1, 10), VxVector(1, 1, 20));

    // Case 1: No transformation (equivalent to AABB test)
    VxMatrix identity;
    Vx3DMatrixIdentity(identity);
    EXPECT_LT(VxIntersect::FrustumOBB(perspectiveFrustum, baseBox, identity), 0.0f) << "OBB with identity matrix inside should be inside.";
    EXPECT_LT(VxIntersect::FrustumBox(perspectiveFrustum, baseBox, identity), 0.0f) << "FrustumBox should yield same result.";


    // Case 2: Rotated but still inside
    VxMatrix rotation;
    Vx3DMatrixFromRotation(rotation, VxVector(0, 1, 0), PI / 4.0f); // 45-degree yaw
    EXPECT_LT(VxIntersect::FrustumOBB(perspectiveFrustum, baseBox, rotation), 0.0f) << "Rotated OBB still inside should be inside.";
    EXPECT_LT(VxIntersect::FrustumBox(perspectiveFrustum, baseBox, rotation), 0.0f) << "FrustumBox should yield same result.";

    // Case 3: Translated to be outside
    VxMatrix translation;
    Vx3DMatrixIdentity(translation);
    translation[3][0] = 50.0f; // Move it far to the right
    EXPECT_GT(VxIntersect::FrustumOBB(perspectiveFrustum, baseBox, translation), 0.0f) << "Translated OBB outside should be outside.";

    // Case 4: A long, thin box rotated to intersect
    VxBbox longBox(VxVector(-100, -1, 15), VxVector(100, 1, 16));
    // When not rotated, it's outside
    EXPECT_GT(VxIntersect::FrustumOBB(perspectiveFrustum, longBox, identity), 0.0f);
    // When rotated 90 degrees, it should intersect
    VxMatrix rotation90;
    Vx3DMatrixFromRotation(rotation90, VxVector(0, 1, 0), PI / 2.0f);
    EXPECT_FLOAT_EQ(VxIntersect::FrustumOBB(perspectiveFrustum, longBox, rotation90), 0.0f) << "Long OBB rotated to intersect should be intersecting.";
}

// Test orthographic frustum classification specifically.
TEST_F(VxFrustumTest, ClassifyPointOrtho) {
    // Point inside the ortho frustum
    VxVector pointInside(0, 0, 0); // Origin is at 0,0,0, frustum is moved back.
    EXPECT_EQ(orthoFrustum.Classify(pointInside), 0);
    EXPECT_TRUE(orthoFrustum.IsInside(pointInside));

    // Point to the left of the ortho frustum
    // Left boundary is at x = -5
    VxVector pointLeft(-6, 0, 0);
    EXPECT_EQ(orthoFrustum.Classify(pointLeft), VXCLIP_LEFT);
    EXPECT_FALSE(orthoFrustum.IsInside(pointLeft));

    // Point to the right of the ortho frustum
    // Right boundary is at x = +5
    VxVector pointRight(6, 0, 0);
    EXPECT_EQ(orthoFrustum.Classify(pointRight), VXCLIP_RIGHT);
    EXPECT_FALSE(orthoFrustum.IsInside(pointRight));

    // Point in front of the near plane
    // Near plane is at z = ortho_origin.z + 1 = -49
    VxVector pointInFront(0, 0, -50);
    EXPECT_EQ(orthoFrustum.Classify(pointInFront), VXCLIP_FRONT);
    EXPECT_FALSE(orthoFrustum.IsInside(pointInFront));
}


// Test the ComputeVertices method.
TEST_F(VxFrustumTest, ComputeVertices) {
    VxVector vertices[8];
    perspectiveFrustum.ComputeVertices(vertices);

    // The 8 vertices are:
    // 0: Near-Top-Left
    // 1: Near-Top-Right
    // 2: Near-Bottom-Left
    // 3: Near-Bottom-Right
    // 4: Far-Top-Left
    // 5: Far-Top-Right
    // 6: Far-Bottom-Left
    // 7: Far-Bottom-Right

    // For our 90-degree FOV frustum at origin, near plane at 1.0:
    // The half-width and half-height at near plane is 1.0.
    EXPECT_VECTORS_NEAR(vertices[0], VxVector(-1, 1, 1));
    EXPECT_VECTORS_NEAR(vertices[1], VxVector(1, 1, 1));
    EXPECT_VECTORS_NEAR(vertices[2], VxVector(-1, -1, 1));
    EXPECT_VECTORS_NEAR(vertices[3], VxVector(1, -1, 1));

    // At the far plane (100.0), half-width and half-height is 100.0
    EXPECT_VECTORS_NEAR(vertices[4], VxVector(-100, 100, 100));
    EXPECT_VECTORS_NEAR(vertices[5], VxVector(100, 100, 100));
    EXPECT_VECTORS_NEAR(vertices[6], VxVector(-100, -100, 100));
    EXPECT_VECTORS_NEAR(vertices[7], VxVector(100, -100, 100));
}

// Test the Transform method.
TEST_F(VxFrustumTest, Transform) {
    VxFrustum frustum = perspectiveFrustum;

    // Create a transformation matrix (e.g., move and rotate the camera)
    VxMatrix worldMatrix;
    Vx3DMatrixFromEulerAngles(worldMatrix, 0, PI/2.0f, 0); // 90-degree yaw
    worldMatrix[3][0] = 10.0f; // Move 10 units right

    // The transform function takes the *inverse* of the world matrix
    VxMatrix invWorldMatrix;
    Vx3DInverseMatrix(invWorldMatrix, worldMatrix);

    frustum.Transform(invWorldMatrix);

    // Verify the new origin
    EXPECT_VECTORS_NEAR(frustum.GetOrigin(), VxVector(10, 0, 0));

    // Verify the new direction (was (0,0,1), now rotated 90 deg around Y-axis)
    EXPECT_VECTORS_NEAR(frustum.GetDir(), VxVector(1, 0, 0));
    
    // Verify the new up vector (should be unchanged by yaw)
    EXPECT_VECTORS_NEAR(frustum.GetUp(), VxVector(0, 1, 0));

    // Verify the new right vector (was (1,0,0), now rotated)
    EXPECT_VECTORS_NEAR(frustum.GetRight(), VxVector(0, 0, -1));

    // The planes should also have been updated implicitly by the transform.
    // The new near plane should be centered at (10,0,0) + (1,0,0)*1 = (11,0,0) with normal (1,0,0)
    VxPlane expectedNearPlane(VxVector(1,0,0), VxVector(11,0,0));
    EXPECT_PLANES_NEAR(frustum.GetNearPlane(), expectedNearPlane);
}

// Test the equality operator
TEST_F(VxFrustumTest, EqualityOperator) {
    VxFrustum frustum1 = perspectiveFrustum;
    VxFrustum frustum2 = perspectiveFrustum;

    EXPECT_TRUE(frustum1 == frustum2);
    
    // Change a property and check for inequality
    frustum2.GetDMax() = 200.0f;
    EXPECT_FALSE(frustum1 == frustum2);
}